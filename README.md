# A3Test - Assertion Augmented Automated Test Case Generation
Developing a BART based model for the unit test cases generation

# How to replicate

#### About the Environment Setup
First of all, clone this repository to your local machine and access the main dir via the following command:

`https://github.com/awsm-research/A3Test`

`cd A3Test`

#### Then, install the python dependencies via the following command:


`pip install transformers`

`pip install torch`

`pip install numpy`

`pip install tqdm`

`pip install pandas`

`pip install tokenizers`

# To download the training and evaluation in our experiments, run the following commands:

# A3Test
## To Train the Assert Augmented model

``python
python training.py -i train.csv -c codePreTrain.csv -po preModel.pth -o model.pth -s src_fm_fc_ms_ff -t test.csv -v eval.csv -pe 7 -ce 8 -a test.txt -q Defect4jTests.txt
``
<details>
           <summary><h3>See Arguments :mag_right:</h3></summary>
 <p>
   
``python   
parser.add_argument("-i", "--trainInput", dest="trainInput", help="Training file for the model")
parser.add_argument("-c", "--codepretrainInput", dest="codepretrainInput", help="Code Pre Training file for the model")
parser.add_argument("-o", "--modelOutputDir", dest="outPath", help="Output Directory Path for the model")
parser.add_argument("-po", "--premodelOutputDir", dest="PreOutPath", help="Pre Training Output Directory Path for the model")
parser.add_argument("-s", "--sourceLabel", dest="sourceLabel", help="Source Label for the train.csv file")
parser.add_argument("-t", "--testInput", dest="testInput", help="Test Input file for the model accuracy")
parser.add_argument("-v", "--valInput", dest="valInput", help="Val Input file for the model accuracy")
parser.add_argument("-e", "--epochs", dest="epochs", help="Epochs for the model")
parser.add_argument("-ce", "--preCodeEpochs", dest="preCodeEpochs", help="Epochs for the model code pre train")
parser.add_argument("-a","--externalTestFile", dest="externalTestFile", help="External Test Files for generating the UTs")
parser.add_argument("-q","--externalTestFileOutput", dest="externalTestFileOutput", help="External Test Files output for generating the UTs")

``
   
</details>



## To Test the Assert Augmented model

```python
python testScript.py -i model.pth -t test.csv -a test.txt -q Defect4jTests.txt
```
<details>
           <summary><h3>See Arguments :mag_right:</h3></summary>
 <p>
   
```python   
parser.add_argument("-i", "--modelInput", dest="modelInput", help="Saved Model file for the testing the script")
parser.add_argument("-t", "--testInput", dest="testInput", help="Test Input file for the model accuracy")
parser.add_argument("-a","--externalTestFile", dest="externalTestFile", help="External Test Files for generating the UTs")
parser.add_argument("-q","--externalTestFileOutput", dest="externalTestFileOutput", help="External Test Files output for generating the UTs")
```
   
</details>

## Post Processing script

``python
python postProcessingScript.py -i Gson_Plbart.txt -o finalOut55.txt -e errors55.txt
``
<details>
           <summary><h3>See Arguments :mag_right:</h3></summary>
 <p>
   
``python   
parser.add_argument("-i", "--input", dest="input", help="Input txt file to begin the post processing with")
parser.add_argument("-o", "--output", dest="output", help="Output file txt file aftet the post processing")
parser.add_argument("-e", "--errorPath", dest="errorLogs", help="Errors Logs ")
``
   
</details>

# AthenaReplication
## To Train the BART model

``python
python training.py -i train.csv -p enPreTrain.csv -c codePreTrain.csv -eo enModel.pth -po preModel.pth -o model.pth -s src_fm_fc_ms_ff -t test.csv -v eval.csv -e 20 -pe 7 -ce 8 -a test.txt -q Defect4jTests.txt
``
<details>
           <summary><h3>See Arguments :mag_right:</h3></summary>
 <p>
   
``python   
parser.add_argument("-i", "--trainInput", dest="trainInput", help="Training file for the model")
parser.add_argument("-o", "--modelOutputDir", dest="outPath", help="Output Directory Path for the model")
parser.add_argument("-s", "--sourceLabel", dest="sourceLabel", help="Source Label for the train.csv file")
parser.add_argument("-t", "--testInput", dest="testInput", help="Test Input file for the model accuracy")
parser.add_argument("-v", "--valInput", dest="valInput", help="Val Input file for the model accuracy")
parser.add_argument("-e", "--epochs", dest="epochs", help="Epochs for the model")
parser.add_argument("-a","--externalTestFile", dest="externalTestFile", help="External Test Files for generating the UTs")
parser.add_argument("-q","--externalTestFileOutput", dest="externalTestFileOutput", help="External Test Files output for generating the UTs")
``
   
</details>



## To Test the BART model

```python
python testScript.py -i model.pth -t test.csv -a test.txt -q Defect4jTests.txt
```
<details>
           <summary><h3>See Arguments :mag_right:</h3></summary>
 <p>
   
```python   
parser.add_argument("-i", "--modelInput", dest="modelInput", help="Saved Model file for the testing the script")
parser.add_argument("-t", "--testInput", dest="testInput", help="Test Input file for the model accuracy")
parser.add_argument("-a","--externalTestFile", dest="externalTestFile", help="External Test Files for generating the UTs")
parser.add_argument("-q","--externalTestFileOutput", dest="externalTestFileOutput", help="External Test Files output for generating the UTs")
```
   
</details>

# PLBART
## To Train the plBart model

``python
python plBartTraining.py -i train.csv -o FinalplBartModelDir -s src_fm_fc_ms_ff -t test.csv -v eval.csv -ce 8 -a test.txt -q Defect4jTests.txt
``
<details>
           <summary><h3>See Arguments :mag_right:</h3></summary>
 <p>
   
``python   
parser.add_argument("-i", "--trainInput", dest="trainInput", help="Training file for the model")
parser.add_argument("-o", "--modelOutputDir", dest="outPath", help="Output Directory Path for the model")
parser.add_argument("-s", "--sourceLabel", dest="sourceLabel", help="Source Label for the train.csv file")
parser.add_argument("-t", "--testInput", dest="testInput", help="Test Input file for the model accuracy")
parser.add_argument("-v", "--valInput", dest="valInput", help="Val Input file for the model accuracy")
parser.add_argument("-e", "--epochs", dest="epochs", help="Epochs for the model")
parser.add_argument("-a","--externalTestFile", dest="externalTestFile", help="External Test Files for generating the UTs")
parser.add_argument("-q","--externalTestFileOutput", dest="externalTestFileOutput", help="External Test Files output for generating the UTs")
``
   
</details>

## To Test the plBart model

``python
python plBartTest.py -i FinalPlBartModelDir -t test.csv -a test.txt -q Defect4jTests.txt
``
<details>
           <summary><h3>See Arguments :mag_right:</h3></summary>
 <p>
   
``python   
parser.add_argument("-i", "--modelInput", dest="modelInput", help="Saved Model file for the testing the script")
parser.add_argument("-t", "--testInput", dest="testInput", help="Test Input file for the model accuracy")
parser.add_argument("-a","--externalTestFile", dest="externalTestFile", help="External Test Files for generating the UTs")
parser.add_argument("-q","--externalTestFileOutput", dest="externalTestFileOutput", help="External Test Files output for generating the UTs")
``
   
</details>

# CodeT5
## To Train the codeT5 model

``python
python codeT5Training.py -i train.csv -o FinalplBartModelDir -s src_fm_fc_ms_ff -t test.csv -v eval.csv -ce 8 -a test.txt -q Defect4jTests.txt
``
<details>
           <summary><h3>See Arguments :mag_right:</h3></summary>
 <p>
   
``python   
parser.add_argument("-i", "--trainInput", dest="trainInput", help="Training file for the model")
parser.add_argument("-o", "--modelOutputDir", dest="outPath", help="Output Directory Path for the model")
parser.add_argument("-s", "--sourceLabel", dest="sourceLabel", help="Source Label for the train.csv file")
parser.add_argument("-t", "--testInput", dest="testInput", help="Test Input file for the model accuracy")
parser.add_argument("-v", "--valInput", dest="valInput", help="Val Input file for the model accuracy")
parser.add_argument("-e", "--epochs", dest="epochs", help="Epochs for the model")
parser.add_argument("-a","--externalTestFile", dest="externalTestFile", help="External Test Files for generating the UTs")
parser.add_argument("-q","--externalTestFileOutput", dest="externalTestFileOutput", help="External Test Files output for generating the UTs")
``
   
</details>



## To Test the codeT5 model

``python
python codeT5Test.py -i FinalModelDir -t test.csv -a test.txt -q Defect4jTests.txt
``
<details>
           <summary><h3>See Arguments :mag_right:</h3></summary>
 <p>
   
``python   
parser.add_argument("-i", "--modelInput", dest="modelInput", help="Saved Model file for the testing the script")
parser.add_argument("-t", "--testInput", dest="testInput", help="Test Input file for the model accuracy")
parser.add_argument("-a","--externalTestFile", dest="externalTestFile", help="External Test Files for generating the UTs")
parser.add_argument("-q","--externalTestFileOutput", dest="externalTestFileOutput", help="External Test Files output for generating the UTs")
``
   
</details>

# CodeBERT
## To Train the codeBERT model

``python
python codeBERTTraining.py -i train.csv -o FinalplBartModelDir -s src_fm_fc_ms_ff -t test.csv -v eval.csv -ce 8 -a test.txt -q Defect4jTests.txt
``
<details>
           <summary><h3>See Arguments :mag_right:</h3></summary>
 <p>
   
``python   
parser.add_argument("-i", "--trainInput", dest="trainInput", help="Training file for the model")
parser.add_argument("-o", "--modelOutputDir", dest="outPath", help="Output Directory Path for the model")
parser.add_argument("-s", "--sourceLabel", dest="sourceLabel", help="Source Label for the train.csv file")
parser.add_argument("-t", "--testInput", dest="testInput", help="Test Input file for the model accuracy")
parser.add_argument("-v", "--valInput", dest="valInput", help="Val Input file for the model accuracy")
parser.add_argument("-e", "--epochs", dest="epochs", help="Epochs for the model")
parser.add_argument("-a","--externalTestFile", dest="externalTestFile", help="External Test Files for generating the UTs")
parser.add_argument("-q","--externalTestFileOutput", dest="externalTestFileOutput", help="External Test Files output for generating the UTs")
``
   
</details>

## To Test the codeBERT model

``python
python codeBERT.py -i FinalModelDir -t test.csv -a test.txt -q Defect4jTests.txt
``
<details>
           <summary><h3>See Arguments :mag_right:</h3></summary>
 <p>
   
``python   
parser.add_argument("-i", "--modelInput", dest="modelInput", help="Saved Model file for the testing the script")
parser.add_argument("-t", "--testInput", dest="testInput", help="Test Input file for the model accuracy")
parser.add_argument("-a","--externalTestFile", dest="externalTestFile", help="External Test Files for generating the UTs")
parser.add_argument("-q","--externalTestFileOutput", dest="externalTestFileOutput", help="External Test Files output for generating the UTs")
``
   
</details>

# CodeGPT
## To Train the codeGPT model

``python
python codeGPTTraining.py -i train.csv -o FinalplBartModelDir -s src_fm_fc_ms_ff -t test.csv -v eval.csv -ce 8 -a test.txt -q Defect4jTests.txt
``
<details>
           <summary><h3>See Arguments :mag_right:</h3></summary>
 <p>
   
``python   
parser.add_argument("-i", "--trainInput", dest="trainInput", help="Training file for the model")
parser.add_argument("-o", "--modelOutputDir", dest="outPath", help="Output Directory Path for the model")
parser.add_argument("-s", "--sourceLabel", dest="sourceLabel", help="Source Label for the train.csv file")
parser.add_argument("-t", "--testInput", dest="testInput", help="Test Input file for the model accuracy")
parser.add_argument("-v", "--valInput", dest="valInput", help="Val Input file for the model accuracy")
parser.add_argument("-e", "--epochs", dest="epochs", help="Epochs for the model")
parser.add_argument("-a","--externalTestFile", dest="externalTestFile", help="External Test Files for generating the UTs")
parser.add_argument("-q","--externalTestFileOutput", dest="externalTestFileOutput", help="External Test Files output for generating the UTs")
``
   
</details>



## To Test the codeGPT model

``python
python codeGPT.py -i FinalModelDir -t test.csv -a test.txt -q Defect4jTests.txt
``
<details>
           <summary><h3>See Arguments :mag_right:</h3></summary>
 <p>
   
``python   
parser.add_argument("-i", "--modelInput", dest="modelInput", help="Saved Model file for the testing the script")
parser.add_argument("-t", "--testInput", dest="testInput", help="Test Input file for the model accuracy")
parser.add_argument("-a","--externalTestFile", dest="externalTestFile", help="External Test Files for generating the UTs")
parser.add_argument("-q","--externalTestFileOutput", dest="externalTestFileOutput", help="External Test Files output for generating the UTs")
``
   
</details>

