# SemAder

---

This repository is the official implementation of  **SemAder: Adversarial Code Generation via Structure-Semantics Joint Induction**.

## Requirement

---

### Run environment

```
Ubuntu + clang-18
```

The test cases we use are Windows-based source code and corresponding IR files.

### Depdence tools

```
Disassembly tool: IDA  
Source code analysis tool: Joern
```

### Pip requirement

```
pip install networkx torch collections numpy tqdm scipy gym faiss pandas gensim openai transformers
```

## Core catalog description

---

```
	+ SemAder
		Main directory for implementation code
		+ work
			Working directory
			+ base
				Test cases
			+ out
				Output directory after applying SemAder
		+ joern-cli
			Directory for the Joern code analysis tool
		+ index
			Induced corpus and corresponding FAISS database index vectors
		+ checkpoints
			Directory for SemAder model checkpoints
		+ obfus_json
			Directory for selecting induced corpus
		+ model
			Directory for the dual-embedding support model
		+ tools
			Directory for disassembly tools
```

## Run

---

To run SemAder and obtain the induced program.

1. Modify the configuration in `config.py`

   Set the name of the original project to be induced, the target function for induction, and the type of induction for the target function.

   Configure the API address (`base_url`) and key (`api_key`) for the evaluation model, and set up the training dataset (if retraining is required).

2. Run SemAder

   ```
   python run.py
   ```


## Result

---

The output directory for the induced executable files (`.exe`) is:

```
SemAder/work/out
```

During successful execution, the working content includes:

<img src=".\Figure\image-20250312103755622.png" alt="image-20250312103755622" style="zoom: 67%;" />
