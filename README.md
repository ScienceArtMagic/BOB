# BOB (Bits, Ordinals, Bytes) and friends

## Why?
Does it really make sense to waste all that time poring over text corpora, breaking it up into subwords, and training them into hundreds of millions of parameters of a big, slow lookup table just for a model that only produces ~3-4 English characters per model pass? And, if you don't tie those parameters to the LM head, to devote hundreds of millions more params for predicting said ~3-4 tokens/pass? BOB doesn't think so.

BOB wraps a language model with nonparametric input and output utilities to replace tokenization completely; without the inefficiency commonly associated with byte-level or character-level modeling (one byte/character per pass), with more character-dense representations than typical pooling or patching setups (e.g. [MegaByte](https://arxiv.org/abs/2305.07185) [[code, LucidRains impl.](https://github.com/lucidrains/MEGABYTE-pytorch)], [SpaceByte](https://arxiv.org/abs/2404.14408) [[code, author impl.](https://github.com/kjslag/spacebyte)]), and produces more characters/pass than byte-/character-level alternatives even when they leverage speculative decoding (e.g. [MambaByte](https://arxiv.org/abs/2401.13660) [[code, author impl.](https://github.com/jxiw/MambaByte)]) or [Multi-token Prediction](https://arxiv.org/abs/2404.19737) (and, being orthogonal, can be combined with such approaches). It also dramatically reduces prefill steps, as input vectors (akin to those you'd get from subword embeddings in a tokenized model) can represent up to `hidden_dim // 4` bytes worth of characters.

## BOB's friends:

### BAT (BOB Anti-Tokenizer)
BOB is quite the contrarian. Why wouldn't BOB have a BAT?

BAT takes input text and produces character ordinals (i.e. the outputs of Python's `ord()` built-in or JavaScipt's `.charCodeAt` property), their bit length (whose usefulness becomes more apparent with BAE), and bytes (as integer representations). Instead of token IDs, BAT hands these off to BAE for on-the-fly (get it?) construction of massively multi-character "token" "embeddings."

### BAE (BOB Anti-Embeddings)
BOB and his BAT get lonely sometimes, but BAE is the perfect companion.

Instead of the usually-massive (`hidden_dim * vocab_size`) "lookup table" (word token embedding matrix) mentioned above, BAE transforms and combines BAT's outputs into model input vectors: 

1. Vertically-stacked horizontal bitmasks of ordinals, increasing by powers of 2 (3/16 of `hidden_dim`)
2. Simple, compact Absolute Positional Encoding (explicit, "coarse"/per "token", 1/16 of `hidden_dim`) based on iterable chunks of ordinals.
3. Bit lengths of each ordinal (and thus, implicitly, fine per-character, and even, technically, explicit per-exact-bit APE - 3/8 of `hidden_dim`)
4. Byte representations (3/8 of `hidden_dim`),
5. Negative adaptive-average-pooled and mean representations of 3 and 4 in their unfilled slots.

BAE then concatenates them all to a single vector and negative-offsets the remaining unfilled slots (to avoid zero inputs, restorable via a simple $`ReLU`$ if/when needed).

The resulting vectors are somewhat similar to those resulting from tokenization + WTE lookup but, other than special "tokens," only need the information available from basic string conversion methods.

### BLAH (BOB Language model Anti-Head)
BOB is a talker. When you buck as many trends as BOB, you've got a lot to say.

BLAH eliminates the need for (yet another) massive `hidden_dim * vocab_size` matrix for token lookup. 

This can be as simple as (and currently is only) a [straight-through estimator](https://arxiv.org/abs/1308.3432) (boolean cast to number for $`x > 0`$ in the forward pass, $`HardTanh(x)`$ where $`min = -1, max = 1`$ in the backward pass) with a reconstruction of every 8 bits to a byte. For multi-byte characters, the two highest bits of the appropriate number of continuation bytes will be prefilled to `[1, 0, ...]` (i.e. 1 `[1, 0, ...]` continuation byte to follow a start byte beginning with `[1, 1, 0, ...]`, 2 for `[1, 1, 1, 0, ...]`, 3 for `[1, 1, 1, 1, 0, ...]`). 

Other methods may be available in the future but, for now, this allows for a verbosity - `hidden_dim` (or `output_dim` if different) divided by 8 - that exceeds common speculative decoding methods (without an additional draft model, draft heads, or early exit - though, again, it is orthogonal to such techniques).

TODO: Investigate the awkwardly-abbreviated [REST (Retrieval-Based Speculative Decoding)](https://arxiv.org/abs/2311.08252). This may be a focus (or at least inspiration) for a future BLAH option.

This byte output is then passed back to BAE to construct new vectors, potentially prefilling multiple continuation vectors per next pass (TODO).

### SISTERS (SImple Special-Token-only Embeddings are Really Small)

BOB and his friends make their own substitute "tokens"/"embeddings," but his SISTERS help with the real deal on special occasions.

No quotes here. Special tokens are still tokens/embeddings just as they are in fully tokenized models, but only in place of (by default/for the time being) 30 of the first 32 UTF-8 codepoints; `\t`/`b\x09`/`9` decimal, a.k.a. "tab" and `\n`/`b\x0a`/`10` decimal a.k.a. "linefeed"/"newline" are combined with other printable characters (you'll have to handle conversion to Windows formatting yourself, if you need carriage returns). All other codepoints from `b\x00`/`0` to `b\x31`/`31` are stored in a `vocab.json`/`tokenizer.json` just like you would expect from a tokenized Hugging Face Transformers-compatible model. Due to the nature of special tokens (and the tiny size of a `hidden_dim * 30` embedding matrix), leaving their representations trainable, and the full `hidden_dim`, makes sense. 

In the future, more special tokens may be made available for the currently unused bytes `b\x80`/128 decimal through `b\x9f`/159 decimal, and possibly some of the many other control characters spread throughout Unicode (though 30 is already a lot, let alone 62).
