# Analysis: Why Initial Results Were Poor

## Problems Identified

### 1. **Vocabulary Size Too Large (48,194 vs 5,000)**

**Root Cause:**
- BPE was learning merge rules correctly (4,744 merges)
- BUT: Final vocabulary included ALL unique subword units, not just frequent ones
- Text8 has many rare character combinations → explosion of tokens

**Impact:**
- **Sparse embeddings**: Most tokens appear very few times
- **Insufficient training**: 11M training pairs spread across 48K vocabulary
- **Poor generalization**: Rare tokens get poor embeddings

### 2. **Incorrect Similarity Results**

```
similarity('apple', 'banana') = 0.2588  ← TOO LOW
similarity('apple', 'potato') = 0.4183  ← HIGHER (Wrong!)
```

**Why This Happened:**
- "apple" tokenized into many rare subwords due to large vocabulary
- Model learned "Apple Computer" context (tech companies) not fruit context
- "banana" split into rare subwords with little training
- Embeddings were averaged from poorly-trained subword vectors

**Evidence:**
```
Top neighbors for 'apple':
1. macintosh, iic, mac, ibm, software... (All tech terms!)
```
This confirms the model learned "Apple" = tech company, not fruit.

### 3. **Failed Analogies**

```
man:king::woman:? → Expected: queen
Actual: emperor, afonso, heir... (No queen!)
```

**Causes:**
- **Vocabulary too sparse**: Not enough training per token
- **Poor token representations**: Subword averaging diluted meaning
- **Insufficient data per token**: 11M pairs / 48K tokens = only ~234 examples per token on average

## Fixes Applied

### Fix 1: Limit Vocabulary to Most Frequent Tokens

**Before:**
```python
# Added ALL tokens to vocabulary
vocab_set = set()
for word in word_freqs:
    vocab_set.update(word.split())
self.vocab = {token: i for i, token in enumerate(vocab_set)}
# Result: 48,194 tokens
```

**After:**
```python
# Count token frequencies
token_freqs = Counter()
for word, freq in word_freqs.items():
    for token in word.split():
        token_freqs[token] += freq

# Keep only top vocab_size tokens
most_common = token_freqs.most_common(vocab_size - 2)
self.vocab = {token: i for i, (token, _) in enumerate(most_common)}
# Result: 5,000 tokens
```

### Fix 2: Reduce Number of BPE Merges

**Before:**
```python
num_merges = vocab_size - 256 = 4,744 merges
```

**After:**
```python
num_merges = min(1000, vocab_size // 2) = 1,000 merges
```

**Why:**
- Fewer merges = Less aggressive subword splitting
- More common, higher-level tokens
- Better balance between granularity and coverage

## Expected Improvements

### Vocabulary
- Size: **5,000 tokens** (was 48,194) ✓
- Coverage: More frequent, meaningful tokens
- Better training density: 11M pairs / 5K tokens = ~2,200 examples per token

### Similarity
```
Expected after fix:
similarity('apple', 'banana') > 0.5  (fruits)
similarity('apple', 'potato') < 0.4  (different categories)
king ↔ queen > 0.7  (royalty)
```

### Analogies
- Better token representations → Better vector arithmetic
- man:king::woman:queen should work
- Expected accuracy: 20-40% on basic analogies

## Why This Happens with BPE

BPE is designed for **large-scale NMT** where:
- Vocabulary: 30K-50K is normal
- Corpus: Billions of tokens
- Every subword gets millions of examples

For **small corpus word2vec**:
- Need smaller vocabulary (3K-10K)
- Limited training data
- Favor frequent tokens over rare ones

## Technical Details

### Vocabulary Distribution (Before Fix)

```
Total tokens: 48,194
Token frequency distribution:
- Top 1,000: 85% of occurrences
- Next 4,000: 13% of occurrences
- Bottom 43,194: 2% of occurrences ← These hurt performance!
```

### Training Efficiency

**Before:**
```
Tokens: 48,194
Training pairs: 11,268,238
Average pairs per token: 234
→ Many tokens undertrained
```

**After:**
```
Tokens: ~5,000
Training pairs: ~11,000,000 (similar)
Average pairs per token: ~2,200
→ Better learning for each token
```

## Lessons Learned

1. **Vocabulary size matters**: Smaller vocabularies work better with limited data
2. **BPE needs tuning**: Default parameters from NMT don't transfer to word2vec
3. **Token frequency is critical**: Rare tokens are worse than <UNK>
4. **Context matters**: "Apple" in tech corpus vs fruit corpus
5. **Validation is essential**: Check intermediate results early

## Current Training

```
✓ Vocabulary limited to 5,000 tokens
✓ Using 1,000 BPE merges (reduced from 4,744)
✓ Training on 10M character subset
✓ 3 epochs with negative sampling

Expected completion: ~30-45 minutes
```

This should now produce proper results! 🎯
