import pandas as pd
import torch
from transformers import BertModel, BertTokenizer
from rdflib import Graph, Literal, Namespace, URIRef
import sys

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Namespace for embeddings in RDF
VEC_NS = Namespace("https://www.openlinksw.com/ontology/vvec#")

def generate_bert_embedding(text):
    # Tokenize and convert to tensors
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Taking the mean of the token embeddings for simplicity
        embeddings = torch.mean(outputs.last_hidden_state, dim=1)
    
    # Reduce the embedding to 512 dimensions if needed (BERT output typically has 768 dimensions)
    if embeddings.shape[1] != 512:
        embeddings = torch.nn.functional.adaptive_avg_pool1d(embeddings.unsqueeze(0), 512).squeeze(0)
    
    return embeddings.squeeze().tolist()

def csv_to_rdf(csv_file, subject_column, embedding_columns):
    # Load the CSV file
    df = pd.read_csv(csv_file)
    
    # Create a graph for RDF triples
    g = Graph()

    # Bind the vvec namespace to the desired prefix
    g.bind("vvec", VEC_NS)
    
    # Process each row in the CSV
    for _, row in df.iterrows():
        # Extract subject
        subject = URIRef(row[subject_column])
        
        # Combine embedding columns into one text
        text_to_embed = " ".join(str(row[col]) for col in embedding_columns)
        
        # Generate the 512-dimension BERT embedding
        embedding = generate_bert_embedding(text_to_embed)
        
        # Convert the embedding to a space-separated string for the RDF triple
        embedding_str = " ".join(map(str, embedding))
        
        # Add the RDF triple to the graph
        g.add((subject, VEC_NS.hasVector, Literal(embedding_str)))
    
    return g

if __name__ == "__main__":
    # Parse command-line arguments
    if len(sys.argv) < 4:
        print("Usage: python bert_embeddings_to_rdf.py <csv_file> <subject_column> <embedding_columns>")
        sys.exit(1)

    csv_file = sys.argv[1]
    subject_column = sys.argv[2]
    embedding_columns = sys.argv[3].split(",")
    
    # Optional: Output filename for the TTL file
    output_file = csv_file.replace(".csv", ".ttl")

    # Create the RDF graph
    rdf_graph = csv_to_rdf(csv_file, subject_column, embedding_columns)

    # Serialize and save the RDF graph to a .ttl file
    with open(output_file, "w") as f:
        f.write(rdf_graph.serialize(format="turtle"))

    print(f"RDF-Turtle file saved as: {output_file}")