from flask import Flask, request, jsonify
import os
import pandas as pd
import json
import csv
import boto3
from botocore.exceptions import NoCredentialsError
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util
import numpy as np
import random
import string
from fuzzywuzzy import fuzz
import mysql.connector
from datetime import datetime
import io

# Define the Flask app
app = Flask(__name__)

# Apply CORS to the app to allow cross-origin requests
CORS(app)

# Preload the T5 model and tokenizer at startup
model_path = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_path, timeout=200)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# AWS S3 configuration
s3 = boto3.client('s3')
BUCKET_NAME = 'aitextbook-results-bucket'

def read_txt_file_from_s3(bucket_name, key):
    """Reads a TXT file from S3 and returns the content as a list of non-empty lines."""
    try:
        print(f"Attempting to read file from S3 bucket '{bucket_name}', key: '{key}'")
        obj = s3.get_object(Bucket=bucket_name, Key=key)
        content = obj['Body'].read().decode('utf-8').splitlines()
        print(f"Successfully read file from S3: {key}")
        return [line.strip() for line in content if line.strip()]
    except Exception as e:
        print(f"Error reading file from S3: {e}")
        return None

def read_csv_file_from_s3(bucket_name, key):
    """Reads a CSV file from S3, prints found columns, and returns the content of the 'questions' column."""
    try:
        print(f"Attempting to read CSV file from S3 bucket '{bucket_name}', key: '{key}'")
        
        # Retrieve the object from S3
        obj = s3.get_object(Bucket=bucket_name, Key=key)
        content = obj['Body'].read().decode('utf-8').splitlines()

        # Use the csv module to read the CSV file
        csv_reader = csv.DictReader(content)

        # Print the found columns (header)
        headers = csv_reader.fieldnames
        print(f"Found columns: {headers}")

        # Extract the 'questions' column
        questions = []
        for row in csv_reader:
            if 'question' in row and row['question'].strip():
                question = row['question']
                questions.append(question)

        # Print all extracted questions at once
        print(f"All extracted questions: {questions}")

        print(f"Successfully read CSV file from S3: {key}")
        return questions
    except Exception as e:
        print(f"Error reading CSV file from S3: {e}")
        return None



# Find the most relevant section of the text for the question
def find_relevant_context(text, question, top_n=1):
    print(f"Finding relevant context for question: '{question}'")
    vectorizer = TfidfVectorizer().fit_transform([question] + text)
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors[0:1], vectors[1:])
    relevant_indices = cosine_sim.argsort()[0][-top_n:]
    relevant_text = " ".join([text[idx] for idx in relevant_indices])
    print(f"Relevant context found: '{relevant_text}'")
    return relevant_text

# Generate predictions and probabilities for a custom context and question
def generate_prediction(context, question):
    print(f"Generating prediction for question: '{question}' with context: '{context}'")
    input_text = f"question: {question} context: {context}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    outputs = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, num_return_sequences=5, num_beams=5, return_dict_in_generate=True, output_scores=True, max_new_tokens=50)

    # Decode predictions and calculate probabilities
    decoded_answers = [tokenizer.decode(seq, skip_special_tokens=True) for seq in outputs.sequences]
    probabilities = F.softmax(outputs.sequences_scores, dim=0).tolist()

    # Combine answers with their probabilities and sort by probability
    answers_with_probabilities = sorted(
        [(answer if answer else 'Could not find an appropriate answer in the given context', prob)
         for answer, prob in zip(decoded_answers, probabilities)],
        key=lambda x: x[1],  # Sort by probability
        reverse=True  # Descending order
    )

    # Select the top 4 answers or all available if less than 4
    top_answers = answers_with_probabilities[:min(4, len(answers_with_probabilities))]

    # Format the answers with probabilities
    formatted_answers = [f"{answer} ({prob:.3f})" for answer, prob in top_answers]
    
    print(f"Generated prediction: '{formatted_answers}'")
    return ", ".join(formatted_answers)

def save_results_to_csv(csv_filename, questions, results):
    """Saves the results to a CSV file."""
    try:
        print(f"Saving results to CSV file: '{csv_filename}'")
        with open(csv_filename, 'w', newline='') as csvfile:
            fieldnames = ['Question', 'Relevant Context', 'Predicted Answer']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for question, result in zip(questions, results):
                writer.writerow({
                    'Question': question,
                    'Relevant Context': result['relevant_context'],
                    'Predicted Answer': result['predicted_answer']
                })
        print(f"CSV file saved: '{csv_filename}'")
        return csv_filename
    except Exception as e:
        print(f"Error saving CSV file: {e}")
        return None

def upload_file_to_s3(file_name, bucket_name, object_name=None):
    """Upload a file to an S3 bucket."""
    if object_name is None:
        object_name = os.path.basename(file_name)

    try:
        print(f"Uploading file '{file_name}' to S3 bucket '{bucket_name}', key: 'uploads/{object_name}'")
        s3.upload_file(file_name, bucket_name, f"uploads/{object_name}")
        print(f"File {file_name} successfully uploaded to S3 as {object_name}.")
    except NoCredentialsError:
        print("Error: AWS credentials not available.")
    except Exception as e:
        print(f"Failed to upload file to S3: {e}")

def write_result_to_s3(outcome, bucket_name, object_name):
    """Writes the outcome to a result.json file in S3."""
    try:
        print(f"Writing result to S3 bucket '{bucket_name}', key: 'uploads/{object_name}'")
        result_data = json.dumps(outcome)
        s3.put_object(Body=result_data, Bucket=bucket_name, Key=f"uploads/{object_name}")
        print(f"Result JSON uploaded to S3 as '{object_name}'.")
    except Exception as e:
        print(f"Failed to upload result.json to S3: {e}")

@app.route('/run', methods=['POST'])
def run():
    try:
        context_filename = request.form['context']
        question_filename = request.form['question']

        print(f"Received context file: '{context_filename}', question file: '{question_filename}'")
        print(f"Context file path: 'uploads/{context_filename}'")
        print(f"Question file path: 'uploads/{question_filename}'")

        # Read content from S3
        context = read_txt_file_from_s3(BUCKET_NAME, f"uploads/{context_filename}")
        questions = read_csv_file_from_s3(BUCKET_NAME, f"uploads/{question_filename}")

        # Debug: Print extracted questions
        print(f"Extracted questions: {questions}")

        if context is None or questions is None or len(questions) == 0:
            outcome = {"message": "Error: One or both files could not be read from S3, or no questions found.", "is_error": True}
            write_result_to_s3(outcome, BUCKET_NAME, 'result.json')
            return jsonify(outcome), 400  # Bad request

        results = []
        # Process questions and generate answers
        for question in questions:
            relevant_context = find_relevant_context(context, question)
            predicted_answer = generate_prediction(relevant_context, question)
            print(f"Processing complete for question: '{question}'")
            results.append({
                "question": question,
                "relevant_context": relevant_context,
                "predicted_answer": predicted_answer
            })

        # Create a CSV file from the results
        csv_filename = f"{os.path.splitext(context_filename)[0]}_results.csv"
        csv_filepath = save_results_to_csv(csv_filename, questions, results)

        if csv_filepath:
            # Upload the CSV file to the S3 bucket
            upload_file_to_s3(csv_filepath, BUCKET_NAME)

            print("Success: Results have been predicted and saved to S3.")
            outcome = {"message": 'Success: Results have been predicted and saved to S3.', "is_error": False}
            write_result_to_s3(outcome, BUCKET_NAME, 'result.json')
            return jsonify(outcome), 200  # OK
        else:
            outcome = {"message": 'Error: Failed to save CSV file.', "is_error": True}
            write_result_to_s3(outcome, BUCKET_NAME, 'result.json')
            return jsonify(outcome), 500  # Internal Server Error
    
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        outcome = {"message": f'Error: {str(e)}', "is_error": True}
        write_result_to_s3(outcome, BUCKET_NAME, 'result.json')
        return jsonify(outcome), 500  # Internal Server Error


@app.route('/practice', methods=['POST'])
def practice():
    try:
        context = request.form['context']
        question = request.form['question']

        if not context or not question:
            outcome = {"message": "Error: Context or question is missing.", "is_error": True}
            return jsonify(outcome), 400  # Bad request

        # Generate prediction for the given context and question
        predicted_answer = generate_prediction(model, tokenizer, context, question)
        
        outcome = {"message": "Success", "is_error": False, "predicted_answer": predicted_answer}
        return jsonify(outcome), 200  # OK
    
    except Exception as e:
        outcome = {"message": f'Error: {str(e)}', "is_error": True}
        return jsonify(outcome), 500  # Internal Server Error

# Load the pre-trained Sentence Transformer model for scoring
scoring_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def generate_scoring_filename(length=10):
    """Generate a random filename for storing scoring results."""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length)) + '_scoring.csv'

def exact_match_score(correct_answer, student_response):
    """Check for exact match of the correct answer in the student's response."""
    return 1 if correct_answer.lower() in student_response.lower() else 0

def fuzzy_matching_score(correct_answer, student_response):
    """Use fuzzy matching to score similarity between correct answer and student response."""
    return fuzz.ratio(correct_answer.lower(), student_response.lower()) / 100.0

def keyword_overlap_score(correct_answer, student_response):
    """Calculate the keyword overlap score between correct answer and student response."""
    correct_keywords = set(correct_answer.lower().split())
    response_keywords = set(student_response.lower().split())
    overlap = correct_keywords & response_keywords
    return len(overlap) / len(correct_keywords)

def score_student_response(question, student_response, correct_answer):
    """Score a student's response using a weighted combination of methods."""
    try:
        print(f"Scoring question: {question}")
        print(f"Correct Answer for scoring: {correct_answer}")

        # Exact Match Score
        exact_score = exact_match_score(correct_answer, student_response)

        # Semantic Similarity Score
        embeddings = scoring_model.encode([student_response, correct_answer])
        semantic_similarity_score = util.cos_sim(embeddings[0], embeddings[1]).item()

        # Fuzzy Matching Score
        fuzzy_score = fuzzy_matching_score(correct_answer, student_response)

        # Keyword Overlap Score
        keyword_score = keyword_overlap_score(correct_answer, student_response)

        # Weighted sum of the scores (final score)
        final_score = (exact_score * 0.6) + (semantic_similarity_score * 0.05) + \
                      (fuzzy_score * 0.3) + (keyword_score * 0.05)

        print(f"Final Score: {final_score:.2f}")

        return final_score, exact_score, fuzzy_score, semantic_similarity_score, keyword_score, student_response, correct_answer
    except Exception as e:
        print(f"Error in scoring question: {e}")
        raise

def save_scoring_results_to_s3(scoring_results_df, scoring_filename):
    """Saves the scoring results to the outputs folder in S3."""
    try:
        output_dir = "outputs"
        output_path = f"{output_dir}/{scoring_filename}"
        print(f"Saving scoring results to S3 bucket '{BUCKET_NAME}', key: '{output_path}'")

        # Convert DataFrame to CSV format in memory
        csv_buffer = io.StringIO()
        scoring_results_df.to_csv(csv_buffer, index=False)

        # Upload CSV directly to S3
        s3.put_object(Body=csv_buffer.getvalue(), Bucket=BUCKET_NAME, Key=output_path)
        print(f"Scoring results successfully saved to S3 at '{output_path}'.")
    except Exception as e:
        print(f"Error saving scoring results to S3: {e}")

@app.route('/score', methods=['POST'])
def score():
    """Endpoint to score student responses."""
    print("Received a POST request to /score")
    
    # Extracting questions and responses from the form data
    questions = []
    responses = []
    student_id = request.form.get('student_id')
    assignment_id = request.form.get('assignment_id')
    
    for key in request.form:
        print(f"Processing form data - Key: {key}, Value: {request.form[key]}")
        if key.startswith('question'):
            # Directly append the question without preprocessing
            questions.append(request.form[key])
        elif key.startswith('response'):
            responses.append(request.form[key])

    # Retrieve entry_code and group_name from the students table
    entry_code = None
    group_name = None
    try:
        connection = mysql.connector.connect(
            host='us-cluster-east-01.k8s.cleardb.net',
            database='heroku_9ce5aebc412feee',
            user='b4cc3b3ad647c9',
            password='25a4fe7f'
        )
        cursor = connection.cursor(dictionary=True)
        
        select_query = """
        SELECT entry_code, group_name FROM students WHERE student_id = %s
        """
        cursor.execute(select_query, (student_id,))
        student_info = cursor.fetchone()
        
        if student_info:
            entry_code = student_info.get('entry_code')
            group_name = student_info.get('group_name')
            print(f"Retrieved entry_code: {entry_code}, group_name: {group_name}")
        else:
            print("Student not found.")
    except mysql.connector.Error as error:
        print(f"Failed to retrieve data from MySQL table: {error}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed.")
    
    # Retrieve topics, difficulty, and correct answers for each question based on assignment_id
    question_info = []
    try:
        connection = mysql.connector.connect(
            host='us-cluster-east-01.k8s.cleardb.net',
            database='heroku_9ce5aebc412feee',
            user='b4cc3b3ad647c9',
            password='25a4fe7f'
        )
        cursor = connection.cursor(dictionary=True)
        
        for question in questions:
            print(f"Searching for question in database: '{question}'")
            select_query = """
            SELECT topic, difficulty, correct 
            FROM multiple_choice_questions 
            WHERE assignment_id = %s AND question = %s
            """
            cursor.execute(select_query, (assignment_id, question))
            question_data = cursor.fetchone()
            
            if question_data:
                print(f"Match found for question '{question}': Topic = {question_data.get('topic')}, Difficulty = {question_data.get('difficulty')}")
                question_info.append({
                    "question": question,
                    "topic": question_data.get('topic'),
                    "difficulty": question_data.get('difficulty'),
                    "correct": question_data.get('correct')
                })
            else:
                print(f"Question '{question}' not found in the database.")
                question_info.append({
                    "question": question,
                    "topic": 'N/A',
                    "difficulty": 'N/A',
                    "correct": 'N/A'

                })
    
    except mysql.connector.Error as error:
        print(f"Failed to retrieve question info from MySQL table: {error}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed.")
    
    # Prepare to save scoring results
    scoring_results = []

    # Score each question
    for index, (question, response) in enumerate(zip(questions, responses)):
        correct_answer = question_info[index]['correct']  # Get the correct answer
        final_score, exact_score, fuzzy_score, semantic_similarity_score, keyword_score, student_response, correct_answer = score_student_response(question, response, correct_answer)
        if final_score is not None:
            scoring_results.append({
                "Question": question,
                "Topic": question_info[index]['topic'],
                "Difficulty": question_info[index]['difficulty'],
                "Student Response": student_response,
                "Correct Answer": correct_answer,
                "Exact Match": exact_score,
                "Fuzzy Match": fuzzy_score,
                "Semantic Similarity": semantic_similarity_score,
                "Keyword Overlap": keyword_score,
                "Score": final_score  # Final weighted score
            })

    # Save the results to a CSV file with a randomly generated filename for scoring
    scoring_filename = generate_scoring_filename()
    scoring_results_df = pd.DataFrame(scoring_results)
    
    # Save directly to S3 outputs folder
    print(f"Attempting to save scoring results to S3 with filename: {scoring_filename}")
    save_scoring_results_to_s3(scoring_results_df, scoring_filename)
    print("Scoring results successfully saved to S3.")

    # Insert into the student_scores table if CSV was successfully created
    try:
        connection = mysql.connector.connect(
            host='us-cluster-east-01.k8s.cleardb.net',
            database='heroku_9ce5aebc412feee',
            user='b4cc3b3ad647c9',
            password='25a4fe7f'
        )
        cursor = connection.cursor()
        
        insert_query = """
        INSERT INTO student_scores (student_id, assignment_id, model_predicted_file, score_file, attempt_date, entry_code, group_name)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        attempt_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute(insert_query, (student_id, assignment_id, "None", scoring_filename, attempt_date, entry_code, group_name))
        connection.commit()
        
        print("Data inserted into student_scores table successfully.")
    except mysql.connector.Error as error:
        print(f"Failed to insert data into MySQL table: {error}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed.")

    # Return the results and the filename
    return jsonify({"scores": scoring_results, "output_file": scoring_filename})

@app.route('/')
def home():
    print("Home route accessed.")
    return "Welcome to the AI Textbook API"

if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(debug=False)
