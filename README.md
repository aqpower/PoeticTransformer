# PoeticTransformer

## Overview

PoeticTransformer is a project focused on the challenging task of poetry generation within the field of Natural Language Processing (NLP). Unlike general text generation tasks, poetry generation demands a higher emphasis on aesthetic quality and semantic consistency. This project introduces a poetry generation model based on the encoder part of the Transformer architecture, specifically aimed at generating Chinese five-character quatrains.

## Features

- **Transformer Encoder**: Utilizes the powerful representation capabilities of the Transformer encoder to generate poetry.
- **Five-Character Quatrains**: Focuses on generating Chinese classical poetry, each poem consisting of several lines, with each line comprising five characters. This format requires a certain rhythm and structural integrity.
- **Dataset**: Trained on a dataset of five-character quatrains from renowned Chinese poets Li Bai, Liu Yuxi, and Li Shangyin. This specialized dataset helps the model to better capture and generate poetry consistent with this format.

## Installation

Clone the repository:
```bash
git clone https://github.com/aqpower/PoeticTransformer.git
cd PoeticTransformer
```

Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the PoeticTransformer model, run the following command:
```bash
python train.py --dataset_path path/to/dataset
```

### Generating Poetry

To generate poetry using a pre-trained model, run:
```bash
python generate.py --model_path path/to/model --start_text "Initial text"
```

### Evaluation

To evaluate the model's performance, use:
```bash
python evaluate.py --model_path path/to/model --test_dataset path/to/test_dataset
```

## Data

The dataset consists of five-character quatrains from poets Li Bai, Liu Yuxi, and Li Shangyin. You can find the dataset in the `data/` directory or specify your own dataset by following the format provided in the examples.

## Model Architecture

The PoeticTransformer model uses only the encoder part of the Transformer architecture. This design leverages the encoder's ability to capture the rich representation needed for poetry generation while focusing on the rhythmic and structural requirements of Chinese five-character quatrains.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue if you encounter any problems or have suggestions for improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- The dataset is sourced from the [Chinese Poetry GitHub repository](https://github.com/chinese-poetry/chinese-poetry).

For further questions or support, please contact [your email address].

---

Enjoy creating beautiful poetry with PoeticTransformer!
