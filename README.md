# BOB (Bytes, Ordinals, Bits) and friends:

## Why?
Does it really make sense to waste all that time poring over text corpora, breaking it up into subwords, and training them into hundreds of millions of parameters of a big, slow lookup table just for a model that only produces ~3-4 English characters per model pass? And if you don't tie those parameters to the LM head, devote hundreds of millions more params for predicting said ~3-4 tokens/pass? BOB doesn't think so.

BOB wraps a language model with nonparametric input and output utilities to replace tokenization completely; without the inefficiency commonly associated with byte-level or character-level modeling (one byte/character per pass), with more character-dense representations than typical pooling or patching setups (e.g. [MegaByte](https://arxiv.org/abs/2305.07185), [SpaceByte](https://arxiv.org/abs/2404.14408)), and produces more characters/pass than byte-/character-level alternatives even when they leverage speculative decoding (e.g. [MambaByte](https://arxiv.org/abs/2401.13660)) or [Multi-token Prediction](https://arxiv.org/abs/2404.19737) (and, being orthogonal, can be combined with such approaches). 

## BOB's friends:

### BAT (BOB Anti-Tokenizer)
BOB is quite the contrarian. Why wouldn't BOB have a BAT?

BAT takes input text and produces character ordinals (i.e. the outputs of Python's `ord()` built-in or JavaScipt's `.charCodeAt[0]` property), their bit length, and bytes (as integer representations).

### BAE (BOB Anti-Embeddings)
BOB and his BAT get lonely sometimes, but BAE is the perfect companion.

Instead of said, usually massive, (`hidden_dim * vocab_size`) "lookup table" (word token embedding matrix), BAE transforms and combines model input vectors on the fly from BAT's outputs (get it?): 

1. Stacked horizontal bitmasks of ordinals vertically by increasing powers of 2,
2. Bit lengths of each ordinal,
3. Byte representations,
4. Pooled combinations thereof
5. Concatenates them all to a single vector
6. Offsetts unfilled slots to avoid zero inputs.

The resulting outputs are vectors somewhat similar to those resulting from tokenization + WTE lookup, but leveraging the information available from basic string conversion methods instead.

### BLAH (BOB Language model Anti-Head)
BOB is a talker. When you buck as many trends as BOB, you've got a lot to say.

BLAH eliminates the need for (yet another) massive `hidden_dim * vocab_size` matrix for token lookup. For now, this is as simple as $`HardTanh(0, 1)`$ with a reconstruction of every 8 bits to a byte. Other methods may be available in the future but, for now, this allows for a verbosity that exceeds speculative decoding methods - without an additional draft model, draft heads, or early exit - though, again, it is orthogonal to such techniques.

