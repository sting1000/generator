<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
***
***
***
*** To avoid retyping too much info. Do a search and replace for the following:
*** github_username, repo_name, twitter_handle, email, project_title, project_description
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This project implement a 2-layer cascade post-processor for Automatic Speech Recognition (ASR) system.
The purpose of post-processor is to format the spoken-style text toward the corresponding written-style text, 
such as *six feet* to *6ft*, *one hundred fifty pounds* to *150lb*. 

Comparing to Rule-based processor, this cascade post-processor is more accurate, flexible and economical.
The processor pipeline contains two main modules: Classifier and Normalizer. 
Classifier assigns label to each token, which is similar to Named Entity Recognition (NER). 
The labeled sequences that require normalization are sent to Normalizer for conversion.
Normalizer employs Neural Network (RNN structure) and learn transform rules from dataset. 
Thus, the interest sequences are converted to written form, and put but to original sentence as output.

With the help of [Huggingface](https://github.com/huggingface/transformers) and [OpenNMT](https://github.com/OpenNMT/OpenNMT-py),
it is easy and fast to try powerful models and complex RNN structure. For Classifier, pretrained models can be chosen from
DistilBert, Bert, XLM, etc. For RNN Normalizer, Bi-LSTM, Transformer, GRU are all possible units to use. Moreover, some
cutting-edge tricks like copy mechanism can be added through the configure file. 
Therefore, this 2-layer cascade post-processor can be customize according to dataset size, language, use cases, etc. 
TBNorm* entity(s). 
These entities are sent to  **Normalizer** for further conversion.


<!-- GETTING STARTED -->
## Getting Started

### Installation
1. Clone the repo
   ```sh
   git clone https://github.com/github_username/repo_name.git
   ```
2. Clone OpenNMT to repository root
   ```sh
   cd repo_name
   git clone https://github.com/OpenNMT/OpenNMT-py.git
   ```
3. Install requirements
   ```sh
   pip install requirements.txt
   ```
### Prepare dataset
In folder `./tools`, we provide code to prepare dataset from 3 difference source

#### Generated
```sh
   git clone https://github.com/github_username/repo_name.git
 ```



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_



<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/github_username/repo_name/issues) for a list of proposed features (and known issues).





