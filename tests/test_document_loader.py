import unittest
from unittest.mock import patch
from utils.document_loader import load_documents
from llama_index.core import Document

class TestDocumentLoader(unittest.TestCase):
    
    def test_load_documents_success(self):
        # Load the actual PDF file without mocking to inspect its content
        file_path = "/Users/anshulsingh/lockedin/mada/coldmail.pdf"
        documents = load_documents(file_path)

        # Print the content of the first document
        if documents:
            print(f"First document content: {documents[0].text}")
            print(f"Number of documents: {len(documents)}")
            print(f"Metadata of first document: {documents[0].metadata}")
        
        # Test if the documents are loaded correctly
        self.assertGreater(len(documents), 0, "No documents loaded.")
        self.assertIsInstance(documents[0], Document)
        self.assertEqual(documents[0].metadata['source'], file_path)

    @patch('utils.document_loader.partition')
    def test_load_documents_failure(self, mock_partition):
        # Mock the partition function to raise an exception
        mock_partition.side_effect = Exception("Failed to partition")
        file_path = "dummy_path.pdf"
        documents = load_documents(file_path)
        self.assertEqual(len(documents), 0)

if __name__ == '__main__':
    unittest.main()
