1. Once you download the zip file, please extract it and open the PROJ1 folder in Visual Studio. 
(If you are not using Visual Studio, we can use any python compatable IDE and run main.py file in the PROJ1 folder. This can be executed using colab and Jupyter as well)
(If you are unable to run a .py file, I have provided a main.ipynb file that can be used to run on environments like colab)

2. Once you open it, simply run the main.py file, this will generate 3 output files. 
(If you face issues in loading the libraries please do a 'pip install' to install regular expression, os, sys, time and nlkt libraries)

3. Please set right parameters to run the code in two settings

i. To run the code for TREC data set the flag testing = "N" and change input_file = "C:/Users/badda/Downloads/proj1/ft911/ft911" (please use your folder path that has the TREC files, with additional 'ft911' in the end).

ii. To run test file set flag testing = "Y" and change input_file = "C:/Users/badda/Downloads/proj1/testdata_phase2.txt" (please use your folder path to your test document path).

4. For using the user interface where you can input a word and get inverted index of the word, please set the flag interface = "Y", this will create a interactive session for both settings mentioned above.

5. For Query processing set flag to "Y". This will generate the vsm_out.txt file and results.txt file.


Below are the output files format for TREC:

 vsm_out.txt file 

TOPIC	DOCUMENT	UNIQUE#	COSINE_VALUE
352	FT911-3246	1	0.20635501613870869
352	FT911-4701	2	0.17646024840664093
352	FT911-437	3	0.1539351678703274
352	FT911-3618	4	0.14210491916124837
352	FT911-3866	5	0.1353021867994525
352	FT911-1087	6	0.13333236982072227
352	FT911-3772	7	0.12440423369143476
352	FT911-2267	8	0.12333332934579527
352	FT911-1174	9	0.11162843261480475
352	FT911-5179	10	0.11015179726268842

 results.txt file


Topic: 352_title
Precision: 0.2777777777777778
Recall: 1.0

Topic: 352_description_title
Precision: 0.2222222222222222
Recall: 1.0

Topic: 352_narrative_title
Precision: 0.08333333333333333
Recall: 1.0


Extra output file: I have also created a vsm_out_all.txt file that has out put for all main query combinations

TOPIC	DOCUMENT	UNIQUE#	COSINE_VALUE
352_title	FT911-3246	1	0.20635501613870869
352_title	FT911-4701	2	0.17646024840664093
352_title	FT911-437	3	0.1539351678703274


Note: Delete all the output files before re-run
