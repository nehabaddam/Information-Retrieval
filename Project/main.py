# Importing Regular Expression and Porter Stemmer libraries
import re
import os
import sys
import time
import math
import numpy as np
import pandas as pd
from operator import itemgetter
from collections import defaultdict
from nltk.stem import PorterStemmer

# This is text parser class that I created to handle all parsing functions
class TextParser_Indexer:
    def __init__(self, stop_words=None):
        # first we intitialize stop words using Tokenizer class
        self.tokenizer = Tokenizer(stop_words)
        # We create instances for Word Dictionary class
        self.word_dict = WordDictionary()
        # We create instances for file Dictionary class
        self.file_dict = FileDictionary()
        # We create instances for forward and inverted index class
        self.indexer = Indexer()

        self.time_taken = 0

    # This is the function that loads the file content and 
    def parse_file(self, file_path, forward_index_file):
        with open(file_path, 'r') as file:
            file_content = file.read()
            # This is the regular expression pattern to find and iterate over matches of the pattern "<DOC>...</DOC>" 
            doc_tag_matches = re.finditer(r'<DOC>(.*?)<\/DOC>', file_content, re.DOTALL)
            # This loop iterates over each document in the file
            for doc_tag_match in doc_tag_matches:
                # get the content between "<DOC>...</DOC>" 
                document_content = doc_tag_match.group(1)
                # Call function parse_document to parse through each document
                self.parse_document(document_content, forward_index_file)
    
    # This function parses each document within the files
    def parse_document(self, document, forward_index_file):
        # Now within the content we have try to find the '<DOCNO>...</DOCNO>' and '<TEXT>...</TEXT>'
        doc_text_match = re.search(r'<DOCNO>(.*?)</DOCNO>.*?<TEXT>(.*?)</TEXT>', document, re.DOTALL)
        if not doc_text_match:
            return

        # doc_tag_match.group(1) has the content between '<DOCNO>...</DOCNO>' tags
        file_doc_id = self.file_dict.map_document_to_id(doc_text_match.group(1))  

        # doc_tag_match.group(2) has the content between '<TEXT>...</TEXT>'
        text_content = doc_text_match.group(2)

        # This function tokenizes the document content into lower case tokens and removes numeric, splits on non-alphanumeric and eliminates stop words. 
        token_stream = self.tokenizer.tokenize(text_content)

        # Now we stem the tokenized words to their root words
        token_stemmed = [self.word_dict.stem(token) for token in token_stream]

        # print(f"\n{file_doc_id}\t{', '.join(map(str, token_stemmed))}")

        # Here the output is stored in dictionaries, we save both the file name with unique file ID and stemmed token/word with their unique ID
        for _, token_stem in zip(token_stream, token_stemmed):
            word_token_id = self.word_dict.map_word_to_id(token_stem)
            self.word_dict.map_word_id_to_doc(word_token_id, file_doc_id)
        
        # Below function uses word_dict and file_dict to create forward and inverted index
        start_time = time.time()
        self.indexer.index_building(file_doc_id, token_stemmed, self.word_dict.word_map_id )
        end_time = time.time()

        self.time_taken = self.time_taken + (end_time - start_time)


    # This function simply saves the file and word dictionaries 
    def save_dictionary(self, output_file, inverted_index_file, forward_index_file):

        word = self.word_dict.word_map_id
        file = self.file_dict.doc_map_id

        combined_dict = {**word, **file}
        # we also combine the word and file dictionaries together and save it in output file parser_output.txt
        with open(output_file, 'w') as output_file:
            for key, value in combined_dict.items():
                output_file.write(f"{key}\t{value}\n")

        
        with open(inverted_index_file, 'w') as output_file:
            for word, documents in self.indexer.inverted_index.items():
                formatted_list =[]
                for doc_id, freq in documents.items():
                    formatted_list.append(f"{doc_id}: {freq};")
                output_file.write(f"{word}\t{' '.join(formatted_list)}\n")  

        with open(forward_index_file, 'w') as output_file:
            for document, words in self.indexer.forward_index.items():
                formatted_list =[]
                for word_id, freq in words.items():
                    formatted_list.append(f"{word_id}: {freq};")
                output_file.write(f"{document}\t{' '.join(formatted_list)}\n")        

        print(f"For indexing program takes {self.time_taken:.2f} seconds.")

        total_index_size = sys.getsizeof(self.indexer.forward_index) +  sys.getsizeof(self.indexer.inverted_index)
        print(f"Total size of index: {total_index_size} bytes with {sys.getsizeof(self.indexer.forward_index)} for forward index and {sys.getsizeof(self.indexer.inverted_index)} for inverted index ")

    def run_interface(self, stop_words, output_file, inverted_index_file):

        # Load forward index from output file
        forward_index = {}
        with open(output_file, 'r') as f:
            for line in f:
                word, word_id = line.strip().split()
                forward_index[word] = int(word_id)
        
        # Load inverted index from inverted index file
        inverted_index = {}
        with open(inverted_index_file, 'r') as f:
            for line in f:
                word_id, posting_list = line.strip().split('\t')
                postings = posting_list.split(';')
                inverted_index[int(word_id)] = {}
                for posting in postings:
                    if posting.strip() != '':
                        doc_id, freq = posting.strip().split(':')
                        inverted_index[int(word_id)][int(doc_id)] = int(freq)
            ps = PorterStemmer()
    
            # User input
            user_input_word = input("Enter a word: ").lower()
            
            # Check if the word is a stop word
            if user_input_word in stop_words:
                print("The word is a stop word.")
                return
            
            # Stem the word
            stemmed_word = ps.stem(user_input_word)
            
            # Check if the stemmed word exists in the forward index
            if stemmed_word not in forward_index:
                print("The word does not exist in the index.")
                return
            
            # Get word ID
            word_id = forward_index[stemmed_word]
            
            # Check if the word ID exists in the inverted index
            if word_id not in inverted_index:
                print("The word ID does not exist in the inverted index.")
                return
            
            # Print posting list information
            
            formatted_list=[]
            for doc_id, freq in inverted_index[word_id].items():
                formatted_list.append(f"{doc_id}: {freq};")
            print(f"Posting list information for word is \n{user_input_word}: {' '.join(formatted_list)}")

class Tokenizer:
    # This is initialization for the Tokenizer class that I have created
    def __init__(self, stop_words=None):
        # Here I use this to get all the stop words that we have into a set, if there are no stop words I simply initialize it as an empty set
        self.stop_words = set(stop_words) if stop_words else set()

    # This is tokenize function, eliminates numbers, converts to lowercase, and extracts word tokens from the input document to tokenize it. After eliminating stop words, it provides the final list of tokens.
    def tokenize(self, document):

        # All characters are converted to lowercase using the document.lower() function, which makes the matching case-insensitive.
        tokens = re.findall(r'\b\w+\b', document.lower()) 
        
        # If the token has no characters that are numbers, it is included. Tokens with numbers on them are filtered away as a result.
        #If the token is not included in the self.stop_words set of stop words, it is included.
        # this hadles the non-aplhanumeric split on punctuations too.
        tokens = [token for token in tokens if not any(char.isdigit() for char in token) and token not in self.stop_words] 
        
        return tokens

class WordDictionary:
    # This is simple initializing function for Word Dictionary
    def __init__(self):
        # Initializing a mapping of document to word_tokens, id-to-word_token, and word_token-to-id dictionaries for the class.
        self.word_map_id = {}
        self.id_map_word = {}
        self.doc_word_mapping = {}
        # We are using existing PorterStemmer function from nlkt to stem words to their roots
        self.stemmer = PorterStemmer()

    # This function Stems a given word using the Porter Stemmer.
    def stem(self, word):
        return self.stemmer.stem(word)

    # This function adds a new word/token to the dictionary and, if it doesn't already exist, gives it a unique numerical ID.
    def map_word_to_id(self, word_token):
        if word_token not in self.word_map_id:
            word_token_id = len(self.word_map_id) + 1
            self.word_map_id[word_token] = word_token_id
            self.id_map_word[word_token_id] = word_token
        else:
            word_token_id = self.word_map_id[word_token]
        return word_token_id

    # This function adds a word ID to the document's mapping, to keep the mapping information
    def map_word_id_to_doc(self, word_token_id, file_doc_id):
        if file_doc_id not in self.doc_word_mapping:
            self.doc_word_mapping[file_doc_id] = set()
        self.doc_word_mapping[file_doc_id].add(word_token_id)

class FileDictionary:
    # Initializing a dictionary to map document IDs to internal document IDs, dictionary to map internal document IDs to the original document IDs and,
    # a counter that keeps track of the current internal document ID
    def __init__(self):
        self.doc_map_id = {}
        self.id_map_doc = {}
        self.counter = 0

    # This function associates the provided file_doc_id with the current internal document ID and vice versa.
    def map_document_to_id(self, file_doc_id):  
        self.counter += 1
        self.doc_map_id[file_doc_id] = self.counter
        self.id_map_doc[self.counter] = file_doc_id
        return self.counter
        
class Indexer:
    # This is simple initializing function for Indexers
    def __init__(self):
        # Initializing 
        self.forward_index = {}
        self.inverted_index = {}


    def index_building(self, file_doc_id, token_stemmed, word_dict_new):
    
    # This gets the word frequency in the document
        word_frequency = {}
        for word in token_stemmed:
            if word in word_frequency:
                word_frequency[word] += 1
            else:
                word_frequency[word] = 1
        
        
        word_frequency_updated = {}

        # Update inverted index in wordID1: docId1: freq in docID1; docId2: freq in docID2;  format
        for word, freq in word_frequency.items():
            if word_dict_new[word] not in self.inverted_index:
                self.inverted_index[word_dict_new[word]] = {}
            self.inverted_index[word_dict_new[word]][file_doc_id] = freq
            word_frequency_updated[word_dict_new[word]] = freq
        
        # Update inverted index in docID1: …; wordIdi: freq in docID1; wordIdi+1: freq in docID1;  format
        self.forward_index[file_doc_id] = word_frequency_updated
     

class Query_Processing:
    def __init__(self, stop_words=None):
        # first we intitialize stop words using Tokenizer class
        self.tokenizer = Tokenizer(stop_words)
        # We create instances for Word Dictionary class
        self.word_dict = WordDictionary()
        # We create instances for file Dictionary class
        self.file_dict = FileDictionary()
        # We create instances for forward and inverted index class
        self.indexer = Indexer()

    @staticmethod
    def calculate_tf_idf_weights(output_file, forward_index_file, inverted_index_file, queries):
        
        tf_idf_weights = defaultdict(dict)  # Dictionary to store TF-IDF weights for terms
        tf_idf_query = defaultdict(dict)
        idf_weights = {}
        N = 0  # Total number of documents in the collection

        # Read forward index file
        with open(forward_index_file, 'r') as f:
            forward_index_lines = f.readlines()

        # Read inverted index file
        with open(inverted_index_file, 'r') as f:
            inverted_index_lines = f.readlines()

        # Read inverted index file
        with open(output_file, 'r') as f:
            output = f.readlines()
        words = output[:len(inverted_index_lines)]
        files = output[len(inverted_index_lines):]
        # Initialize an empty dictionary
        word_count_dict = {}
        word_to_num_dict = {}
        for line in words:
            parts = line.split('\t')
            word = parts[0]
            count = int(parts[1])  
            word_count_dict[count] = word
            word_to_num_dict[word] = count

        file_count_dict = {}
        for line in files:
            parts = line.split('\t')
            file = parts[0]
            num = int(parts[1])  
            file_count_dict[num] = file


        N = len(forward_index_lines)

        # Calculate IDF for each term in the collection based on the inverted index
        for line in inverted_index_lines:
            term_id, postings = line.strip().split('\t')
            df = len(postings.rstrip(';').split(';'))  # Document frequency (df) of the term
            idf = math.log(N / df) if df > 0 else 0  # IDF calculation

            idf_weights[int(term_id)] = idf
        
        for line in forward_index_lines:    
            file_id, term = line.strip().split('\t')
            term_frequency = term.rstrip(';').split(';')
            for item in term_frequency:
                term_id, tf = item.rstrip(';').split(':')
                tf_idf_weights[word_count_dict.get(int(term_id), 0)][file_count_dict.get(int(file_id), 0)] = int(tf) * idf_weights.get(int(term_id), 0)

        tf_idf_table = pd.DataFrame(tf_idf_weights).T


        for key, item in queries.items():
            for i, j in item.items():
                for k in set(j):
                    if str(k) in tf_idf_weights: 
                        term_id = word_to_num_dict.get(str(k), 0)
                        tf_idf_weights[str(k)][f"{key}_{i}"] = int(j.count(k)) * idf_weights.get(int(term_id), 0)
                    else:
                        tf_idf_weights[str(k)] = {}
                        term_id = word_to_num_dict.get(str(k), 0)
                        tf_idf_weights[str(k)][f"{key}_{i}"] = int(j.count(k)) * idf_weights.get(int(term_id), 0)

        tf_idf_query = pd.DataFrame(tf_idf_weights).T


        return tf_idf_query[tf_idf_table.columns] , tf_idf_query.drop(tf_idf_table.columns, axis=1)

    
    @staticmethod
    def parse_topics_file(topics_file, stop_words):
        extracted_values = {}
        num_pattern = r'<num> Number:\s*(\d+)'
        title_pattern = r'<title>\s*(.*?)\s*<desc>'
        desc_pattern = r'<desc> Description:\s*(.*?)\s*<narr>'
        narr_pattern = r'<narr> Narrative:\s*(.*?)\s*$'
        with open(topics_file, 'r') as file:
            file_content = file.read()
        
        doc_tag_matches = re.finditer(r'<top>(.*?)<\/top>', file_content, re.DOTALL)
        
        for idx, doc_tag_match in enumerate(doc_tag_matches, 1):
            # get the content between "<top>...</top>" 
            document_content = doc_tag_match.group(1)
            doc_values = {}
            ps = PorterStemmer()

            # Extract document number
            num_match = re.search(num_pattern, document_content)
            num = num_match.group(1) if num_match else ""

            # Extract title
            title_match = re.search(title_pattern, document_content, re.DOTALL)
            t = title_match.group(1).strip() if title_match else ""
            title = re.findall(r'\b\w+\b', t.lower()) 
            title =  [token for token in title if not any(char.isdigit() for char in token) and token not in stop_words] 
            main_title = [ps.stem(word) for word in title]

            # Extract description
            desc_match = re.search(desc_pattern, document_content, re.DOTALL)
            d = desc_match.group(1).strip().replace('\n', '') if desc_match else ""
            description = re.findall(r'\b\w+\b', d.lower()) 
            description =  [token for token in description if not any(char.isdigit() for char in token) and token not in stop_words] 
            main_description =  [ps.stem(word) for word in description]
            
            # Extract narrative
            narr_match = re.search(narr_pattern, document_content, re.DOTALL)
            n = narr_match.group(1).strip() if narr_match else ""
            narrative = re.findall(r'\b\w+\b', n.lower())
            narrative =  [token for token in narrative if not any(char.isdigit() for char in token) and token not in stop_words] 
            main_narration = [ps.stem(word) for word in narrative]

            doc_values['title'] = main_title
            doc_values['description_title'] = main_title + main_description
            doc_values['narrative_title'] =  main_title + main_narration

            # Store extracted values with document ID as key
            extracted_values[num] = doc_values

        return extracted_values
    
    @staticmethod
    def compare_queries_to_documents(tf_idf_query, tf_idf_table, output_file_path, relevancy_path):
        similarities = defaultdict(dict)
         
        sorted_similarities= defaultdict(dict)
        # Iterate through each query column
        for query_column in tf_idf_query.columns:
            query_vector = tf_idf_query[query_column].values
            
            unique_counter = 0
            
            # Dictionary to store document similarities for each query
            document_similarities = {}
            
            # Iterate through each document column
            for document_column in tf_idf_table.columns:
                document_vector = tf_idf_table[document_column].values
                
                # Calculate cosine similarity between query and document vectors
                cosine_similarity_score = Query_Processing.cosine_similarity(query_vector, document_vector)

                similarities[query_column][document_column] = cosine_similarity_score


            for query, documents in similarities.items():
                # Sort documents based on cosine similarity score
                sorted_documents = dict(sorted(documents.items(), key=itemgetter(1), reverse=True))
                sorted_similarities[query] = sorted_documents

            file_path = "vsm_output_all.txt"
            
            # Open the files in write mode
            with open(file_path, 'w') as file, open(relevancy_path, 'w') as rel, open(output_file_path, 'w') as out:
                file.write("TOPIC\tDOCUMENT\tUNIQUE#\tCOSINE_VALUE\n")
                rel.write("TOPIC\tITERATION\tDOCUMENT\tRELEVANCY\n")
                out.write("TOPIC\tDOCUMENT\tUNIQUE#\tCOSINE_VALUE\n")

                # Iterate over each topic and its documents
                for topic, documents in sorted_similarities.items():
                    UNIQUE = 0
                    for document, cosine_value in documents.items():
                        # Write data to file in the specified format
                        UNIQUE += 1
                        file.write(f"{topic}\t{document}\t{UNIQUE}\t{cosine_value}\n")
                        if cosine_value > 0:
                            relevancy = 1
                        else:
                            relevancy = 0
                        rel.write(f"{topic}\t0\t{document}\t{relevancy}\n")

                        if "_title" in topic and "_description" not in topic and "_narrative" not in topic:
                            out.write(f"{''.join(filter(str.isdigit, topic))}\t{document}\t{UNIQUE}\t{cosine_value}\n")
                
        return sorted_similarities



    @staticmethod
    def cosine_similarity(query_vector, document_vectors):
        # Calculate dot product between query vector and document vectors
        query_vector = np.nan_to_num(query_vector)
        document_vectors = np.nan_to_num(document_vectors)
        dot_products = np.dot(document_vectors, query_vector)
        
        # Calculate magnitudes of query vector and document vectors
        query_magnitude = np.linalg.norm(query_vector)
        document_magnitudes = np.linalg.norm(document_vectors)
        
        # Avoid division by zero
        if document_magnitudes == 0 or query_magnitude==0:
            document_magnitudes=1
            query_magnitude = 1
        
        # Calculate cosine similarity scores
        cosine_similarities = dot_products / (query_magnitude * document_magnitudes)
        
        return cosine_similarities

    @staticmethod
    def relevancy(mqrels_path, relevancy_path):
            mqrels = {}
            with open(mqrels_path, 'r') as mqrels_file:
                for line in mqrels_file:
                    topic, _, document, relevance = line.strip().split()
                    mqrels.setdefault(topic, {}).setdefault(document, int(relevance))

            # Read relevancy file
            relevancy = {}
            with open(relevancy_path, 'r') as relevancy_file:
                next(relevancy_file)
                for line in relevancy_file:
                    topic, _, document, relevance = line.strip().split()
                    relevancy.setdefault(topic, {}).setdefault(document, int(relevance))

            # Compute precision and recall for each topic and document
            results = {}
            for topic_o in relevancy.keys():
                tp = fp = fn = 0
                for document, rel in relevancy[topic_o].items():
                    topic = ''.join(filter(str.isdigit, topic_o)) 
                    if document in mqrels.get(topic, {}):
                        mqrels_relevance = mqrels[topic][document]
                        relevance = int(rel)
                        if mqrels_relevance == relevance:
                            tp += 1
                        elif mqrels_relevance == 1 and relevance == 0:
                            fn += 1
                        elif mqrels_relevance == 0 and relevance == 1:
                            fp += 1
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                results[topic_o] = {'precision': precision, 'recall': recall}
                
            return results


# The actual code starts running from here. This is the main function
def main(stop_file, input_file, output_file, forward_index_file, inverted_index_file , testing, interface, start_time,query_processing,qrels_file):
    

    # Initially I am simply load all the stopwords from the given file. 
    with open(stop_file, 'r') as file:
        stop_words = set(line.strip() for line in file.read().splitlines())

    # creating an instance for the Text parser.
    parser_instance = TextParser_Indexer(stop_words=stop_words)

    # loading all files with given input file path + _i here i is all numbers from 1 to 15.
    if testing == "Y" and query_processing !="Y":

        parser_instance.parse_file(input_file, forward_index_file)
        
        # Saving the extra output of word and file dictionaries for all these files
        parser_instance.save_dictionary(output_file, inverted_index_file, forward_index_file)

    elif not os.path.exists(output_file):
        files = [f"_{i}" for i in range(1, 16)]

        # For each file, I parse them individually one after the another. Here I am simply passing file names to parse_file function
        for file in files:
            parser_instance.parse_file(input_file + file, forward_index_file)

        # Saving the extra output of word and file dictionaries for all these files
        parser_instance.save_dictionary(output_file, inverted_index_file, forward_index_file)

    # Example usage:
    if query_processing == "Y" :
        relevancy ="N"
        input_file = "C:/Users/badda/Downloads/proj1/topics.txt"
        output_file_path = 'vsm_output.txt'
        relevancy_path = 'relevancy.txt'
        mqrels_path = "main.qrels"
        results_path = "results.txt"

        query_instance = Query_Processing(stop_words=stop_words)
        
        if relevancy == "N":
            queries = query_instance.parse_topics_file(input_file, stop_words)
            tf_idf_table,tf_idf_query = query_instance.calculate_tf_idf_weights(output_file, forward_index_file, inverted_index_file, queries)
            similarities = query_instance.compare_queries_to_documents(tf_idf_query, tf_idf_table, output_file_path, relevancy_path)
        
        results = query_instance.relevancy(mqrels_path, relevancy_path)

        with open(results_path, 'w') as results_file:
            # Iterate over each entry in the dictionary
            for topic, metrics in results.items():
                # Write topic name
                results_file.write(f"Topic: {topic}\n")
                # Write precision and recall values
                results_file.write(f"Precision: {metrics['precision']}\n")
                results_file.write(f"Recall: {metrics['recall']}\n\n")

    print(f"The total time taken to run the entire program including Tokenization and Indexing is {(time.time() - start_time)} seconds")

    if interface == "Y":
        parser_instance.run_interface(stop_words, output_file, inverted_index_file)
       


#This runs the whole code. 
if __name__ == "__main__":

    start_time = time.time()

    # set this flag if you want to run test file
    testing = "N"
    interface = "N"
    query_processing ="Y"

    # input variables
    input_file = "C:/Users/badda/Downloads/proj1/ft911/ft911"
    stop_file = "stopwordlist.txt"
    output_file = "parser_output.txt" 
    forward_index_file = "forward_index.txt" 
    inverted_index_file = "inverted_index.txt" 
    qrels_file = "main.qrels"

    #run main for TREC data 
    if testing != "Y":
        
        main(stop_file, input_file, output_file, forward_index_file, inverted_index_file , testing, interface, start_time, query_processing,qrels_file)
        
    elif testing == "Y":
        input_file = "C:/Users/badda/Downloads/proj1/testdata_phase2.txt"
        main(stop_file, input_file, f"test_{output_file}", f"test_{forward_index_file}" , f"test_{inverted_index_file}", testing, interface, start_time, query_processing,qrels_file)

