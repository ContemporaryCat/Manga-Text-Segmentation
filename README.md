# Manga-Text-Segmentation
Segmentation of text in manga.

## Example
Input:
<img width="1654" height="1170" alt="image" src="https://github.com/user-attachments/assets/251a8aa0-f3bd-4f79-953e-067f777fad0d" />
Output:
<img width="1654" height="1170" alt="image" src="https://github.com/user-attachments/assets/43e17f96-323c-441a-ba57-06ba019a0f8f" />
<sup>(source: [manga109](http://www.manga109.org/en/), © Tadashi Satō)</sup>

## Dataset
Label masks used for training are available at [zenodo](https://zenodo.org/record/4511796). For the original Manga109 images, please refer to [Manga 109 website](http://www.manga109.org/en/).

## Citations:
````
@InProceedings{10.1007/978-3-030-67070-2_38,
author="Del Gobbo, Juli{\'a}n
and Matuk Herrera, Rosana",
editor="Bartoli, Adrien
and Fusiello, Andrea",
title="Unconstrained Text Detection in Manga: A New Dataset and Baseline",
booktitle="Computer Vision -- ECCV 2020 Workshops",
year="2020",
publisher="Springer International Publishing",
address="Cham",
pages="629--646",
abstract="The detection and recognition of unconstrained text is an open problem in research. Text in comic books has unusual styles that raise many challenges for text detection. This work aims to binarize text in a comic genre with highly sophisticated text styles: Japanese manga. To overcome the lack of a manga dataset with text annotations at a pixel level, we create our own. To improve the evaluation and search of an optimal model, in addition to standard metrics in binarization, we implement other special metrics. Using these resources, we designed and evaluated a deep network model, outperforming current methods for text binarization in manga in most metrics.",
isbn="978-3-030-67070-2"
}

@dataset{segmentation_manga_dataset,
  author       = {julian del gobbo and
                  Rosana Matuk Herrera},
  title        = {{Mask Dataset for: Unconstrained Text Detection in 
                   Manga: a New Dataset and Baseline}},
  month        = feb,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {1.0},
  doi          = {10.5281/zenodo.4511796},
  url          = {https://doi.org/10.5281/zenodo.4511796}
}
````
