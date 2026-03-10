from utils.regression_head import RegressionHead
from transformers.utils.hub import cached_file
from utils.embedder import get_embedder_instance
import torch
from fire import Fire
from itertools import islice
import json
import sys
from safetensors.torch import save_file
import psutil
import os


def _docit(maxlen=None):
    for line in sys.stdin:
        data = json.loads(line)
        if maxlen is not None:
            data['text'] = data['text'][:maxlen]
        yield data['id'], data['text']


def _batched(iterable, n):
    it = iter(iterable)
    while True:
        batch = tuple(islice(it, n))
        if not batch:
            return
        yield batch


def print_mem():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()

    print(f"RSS (resident set size): {mem_info.rss / 1024 / 1024:.2f} MB", file=sys.stderr)
    print(f"VMS (virtual memory size): {mem_info.vms / 1024 / 1024:.2f} MB", file=sys.stderr)
    print(f"Memory usage: {process.memory_percent():.2f}%", file=sys.stderr)


class JQLRunner:
    def __init__(self):
        # load embedder
        self.device = 'cuda'
        self.embedder = get_embedder_instance('Snowflake/snowflake-arctic-embed-m-v2.0', self.device, torch.bfloat16)
        # load JQL Edu annotation heads
        regression_head_checkpoints = {
                f'{teacher}-sf-{trainset[0]}': cached_file('Jackal-AI/JQL-Edu-Heads', f'checkpoints/edu-{teacher}-snowflake-{trainset}.ckpt')
            # for teacher in 'gemma'.split() for trainset in 'balanced'.split()
            for teacher in 'gemma llama mistral'.split() for trainset in 'balanced unbalanced'.split()
        }
        self.regression_heads = {}
        for name, path in regression_head_checkpoints.items():
            self.regression_heads[name] = RegressionHead.load_from_checkpoint(path, map_location=self.device).to(torch.bfloat16)
        print('Heads loaded:', self.regression_heads.keys(), file=sys.stderr)


    def test(self):
        # Given a single document
        doc = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua'
        embeddings = self.embedder.embed([doc, doc])
        print(embeddings.shape, embeddings.dtype)
        scores = {}
        with torch.no_grad():
           for name, regression_head in self.regression_heads.items():
            scores[f'score_{name}'] = regression_head(embeddings).cpu().squeeze(1)

        print(scores)




    def process_batched(self, bs, maxlen=None):
        for batch in _batched(_docit(maxlen), bs):
            embeddings = self.embedder.embed([text for idx,text in batch])
            scores = {}
            with torch.no_grad():
               for name, regression_head in self.regression_heads.items():
                scores[name] = regression_head(embeddings).cpu().squeeze(1)
            yield batch, scores, embeddings


    def onlyscore(self, bs, maxlen=None):
        for batch, scores, embeddings in self.process_batched(bs, maxlen):
            for i, (idx,_) in enumerate(batch):
                print(json.dumps({'id':idx, 'JQL': {k:v[i].item() for k,v in scores.items()}}))


    def vectorize_score(self, fout, bs, maxlen=None):
        print_mem()
        print('Running inference...', file=sys.stderr)

        outputs_list = []
        for batch, scores, embeddings in self.process_batched(bs, maxlen):
            outputs_list.append(embeddings.cpu())
            for i, (idx,_) in enumerate(batch):
                print(json.dumps({'id':idx, 'jql': {k:v[i].item() for k,v in scores.items()}}))

        print_mem()
        print('Concatenating and dumping embeddings...', file=sys.stderr)
        # NB! Output tensors are bfloat16, torch save_file should work smoothly with this format, but doesn't support streaming.
        # We'd like to avoid concatenation in memory while streaming output tensors to a single file,
        # it may be possible with h5py, but it doesn't support bfloat16, so we need to figure out and test proper convertions.
        # While we have enough RAM for the input number of docs stick to torch.cat+save_file.
        embs = torch.cat(outputs_list)
        save_file({"embeddings": embs}, f"{fout}.safetensors")
#        save_file({"embeddings": embs[:,:256].contiguous()}, f"{fout}.truncated.safetensors")
        print_mem()
        print('All done', file=sys.stderr)


Fire(JQLRunner)
