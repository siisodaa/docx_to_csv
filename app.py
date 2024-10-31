from flask import Flask
import docx
import csv

app = Flask(__name__)

@app.route('/')

def extract_first_table(docx_file, csv_file):
    # Load the .docx document
    doc = docx.Document(docx_file)

    # Get the first table from the document
    table = doc.tables[0]

    # Extract data from the table
    data = []
    for row in table.rows:
        row_data = [cell.text.strip() for cell in row.cells]
        data.append(row_data)

    # Save the data to a CSV file
    with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(data)

    print(f"First table has been saved to {csv_file}")



if __name__ == "__main__" :
    app.run(debug=True)
