# FOL2NS: Generating Natural Language Sentences from First-order Logic

**Coursework for Directed Reading 2**

---

## Project Overview

This repository contains code and resources for generating FOL formulas with gradual quantifier depth as well as converting them into natural sentences. It has two modules in data construction and applies T5 models for the final translation. The performance metrics are edit distance and BLEU score, reporting several datasets on this logic-to-text task.

---

## Repository Structure

```
root/
│
├── Data Construction/                                               # Dataset
│   └── Preprocessing Dataset.ipynb                                  # └─ Notebook: Preprocessing and constructing datasets
│       ├── Step 1: Training and Validation dataset                  # └─└─ (Module A) Preprocess the FOLIO dataset
│       │    ├── FOLIO_train.json                                    # └─└─└─ The training dataset of FOLIO
│       │    ├── FOLIO_validation.json                               # └─└─└─ The validation dataset of FOLIO
│       │    └── *FOL2NS.json                                        # └─└─└─ The final result for Step 1: the preprocessed dataset for training T5 from the above two datasets
│       │
│       └── Step 2: Test dataset                                     # └─└─ (Module B) Generate FOL formulas 
│            ├── Defined_FOL.json                                    # └─└─└─ Define FOL by two CFG rules
│            └── *FOL2NW_with_predicate.json                         # └─└─└─ The final result for Step 2: Lexicalise Defined FOL to logical form with natural words (predicates-entity pairs) for training T5
│
├── Main experiments/                                                # Fine-tuning the T5 model on FOL2NS
│   ├── 1. Optimal T5 for FOL2NS Translation.ipynb                   # └─ Notebook: The best and final model
│   │        ├── FOL2NS.json                                         # └─└─ Dataset For Training and Validation
│   │        ├── FOL2NW_with_predicate.json                          # └─└─ Dataset For Test
│   │        └── *T5_Defined_FOL2NS.json                             # └─└─ The final result for FOL2NS
│   │
│   └── 2. Hyperparameter Sensitivity for Finetuning T5.ipynb        # └─ Notebook: Find the best configuration among different hyperparameters
│            └── FOL2NS.json                                         # └─└─ Dataset For Training and Validation
│
└── README.md                                                        # Project overview

```
## Running the code
1. For constructing the dataset:
- 1-1. You need to download "FOLIO_train.json" and "FOLIO_validation.json"
- 1-2. Then you can directly run the script of "Preprocessing Dataset.ipynb" in Jupyter notebook
2. For main experiments:
- 2-1. You need to download "FOL2NS.json" and "FOL2NW_with_predicate.json"
- 2-2. You don't need to run the previous step for preprocessing data
- 2-3. Then you can directly run the script of "Optimal T5 for FOL2NS Translation.ipynb" in Jupyter notebook
- 2-4. If you want to try different hyperparameters during training the model, you also need to download "FOL2NS.json" and "FOL2NW_with_predicate.json". Then you can directly run the script of "Hyperparameter Sensitivity for Finetuning T5.ipynb"

## The optimal configuration for T5

### T5 (Seq2Seq)

python Optimal T5 for FOL2NS Translation.ipynb
  --model_type T5-large 
  --prefix: "Translate FOL formulas to English:" 
  --batch_size 8 
  --learning_rate 1e-4 
  --num_epochs 20 
  --Other strategies:
  ----Add special symbols to T5Tokenizer
  ----Replace special symbols with corresponding natural words
  ----In validation stage, use "model.generate()" with "num_beams=5, repetition_penalty=1, no_repeat_ngram_size=2,max_length=64, early_stopping=True
  ----In test stage, use "model.generate()" with "num_beams=5, repetition_penalty=1, no_repeat_ngram_size=3, early_stopping=True, max_length=100"
  --different hyperparameters can be explored\
```

## Citation & References

* Chen, Z., Chen, W., Zha, H., Zhou, X., Zhang, Y., Sundaresan, S., & Wang, W. Y. (2020). Logic2Text: High-fidelity natural language generation from logical forms. Findings of the Association for Computational Linguistics: EMNLP 2020, 2096–2111.
* Tian, J., Li, Y., Chen, W., Xiao, L., He, H., & Jin, Y. (2021). Diagnosing the first-order logical reasoning ability through LogicNLI. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (pp. 3738–3747). Association for Computational Linguistics.
* Han, S., Schoelkopf, H., Zhao, Y., Qi, Z., Riddell, M., Zhou, W., Coady, J., Peng, D., Qiao, Y., Benson, L., Sun, L., Wardle-Solano, A., Szabó, H., Zubova, E., Burtell, M., Fan, J., Liu, Y., Wong, B., Sailor, M., Ni, A., Nan, L., Kasai, J., Yu, T., Zhang, R., Fabbri, A. R., Kryscinski, W. M., Yavuz, S., Liu, Y., Lin, X. V., Joty, S., Zhou, Y., Xiong, C., Ying, R., Cohan, A., & Radev, D. (2024). FOLIO: Natural language reasoning with first-order logic. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (pp. 22017–22031). Association for Computational Linguistics.
* Lalwani, A., Kim, T., Chopra, L., Hahn, C., Jin, Z., & Sachan, M. (2025). Autoformalizing natural language to first-order logic: A case study in logical fallacy detection. arXiv preprint arXiv:2405.02318.
* Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., & Liu, P. J. (2023). Exploring the limits of transfer learning with a unified text-to-text Transformer. arXiv preprint arXiv:1910.10683.
* Levenshtein Vladimir I. (1966). Binary codes capable of correcting deletions, insertions, and reversals. Soviet Physics. Doklady, 10(8), 707–710.
* Papineni, K., Roukos, S., Ward, T., & Zhu, W.-J. (2002). BLEU: A method for automatic evaluation of machine translation. In Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics (pp. 311–318). Association for Computational Linguistics.
* ACL LaTeX template: [https://github.com/acl-org/acl-style-files](https://github.com/acl-org/acl-style-files)
