import json
from googleapiclient.discovery import build
from google.oauth2 import service_account
import time
from tqdm import tqdm
import os
from googleapiclient.http import MediaFileUpload

def insert_image_from_url(service_docs, document_id, image_url, location_end_index):
    requests = [
        {
            'insertInlineImage': {
                'location': {
                    'index': location_end_index,
                },
                'uri': image_url,
                'objectSize': {
                    'height': {'magnitude': 50, 'unit': 'PT'},
                    'width': {'magnitude': 50, 'unit': 'PT'}
                }
            }
        }
    ]
    service_docs.documents().batchUpdate(documentId=document_id, body={'requests': requests}).execute()

def load_json_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def setup_google_docs_api(credentials_path, scopes):
    credentials = service_account.Credentials.from_service_account_file(
        credentials_path, scopes=scopes)
    return build('docs', 'v1', credentials=credentials)

def create_document(service_drive, service_docs, title):
    query = f"name = '{title}' and mimeType = 'application/vnd.google-apps.document'"

    response = service_drive.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    files = response.get('files', [])


    if files:
        return files[0].get('id')
    document_body = {'title': title}
    document = service_docs.documents().create(body=document_body).execute()
    return document.get('documentId')


def insert_text(service, document_id, text, indent_amount=0):
    document = service.documents().get(documentId=document_id).execute()
    document_content = document['body']['content']

    if len(document_content) > 1:
        end_index = document_content[-1]['endIndex'] - 1
    else:
        end_index = 1

    requests = [
        {
            'insertText': {
                'location': {
                    'index': end_index
                },
                'text': '\n' + text
            }
        }
    ]

    bullet_start_index = end_index + 1
    bullet_end_index = bullet_start_index + len(text)
    requests.append({
        'createParagraphBullets': {
            'range': {
                'startIndex': bullet_start_index,
                'endIndex': bullet_end_index
            },
            'bulletPreset': 'NUMBERED_DECIMAL_ALPHA_ROMAN'
        }
    })

    if indent_amount > 0:
        requests.append({
            'updateParagraphStyle': {
                'range': {
                    'startIndex': bullet_start_index,
                    'endIndex': bullet_end_index
                },
                'paragraphStyle': {
                    'indentStart': {
                        'magnitude': indent_amount,
                        'unit': 'PT'
                    }
                },
                'fields': 'indentStart'
            }
        })

    service.documents().batchUpdate(
        documentId=document_id, body={'requests': requests}).execute()

def upload_image_to_drive(service_drive, file_path, folder_id):
    file_metadata = {
        'name': file_path.split('/')[-1],
        'parents': [folder_id]
    }
    media = MediaFileUpload(file_path, mimetype='image/png')
    file = service_drive.files().create(body=file_metadata, media_body=media, fields='id').execute()
    return file.get('id')

def insert_image(service_docs, document_id, image_id):
    document = service_docs.documents().get(documentId=document_id).execute()
    document_content = document['body']['content']

    if len(document_content) > 1:
        end_index = document_content[-1]['endIndex'] - 1
    else:
        end_index = 1

    requests = [{
        'insertInlineImage': {
            'location': {
                'index': end_index
            },
            'uri': f"https://drive.google.com/uc?id={image_id}",
            'objectSize': {
                'height': {'magnitude': 150, 'unit': 'PT'},
                'width': {'magnitude': 150, 'unit': 'PT'}
            }
        }
    }]

    service_docs.documents().batchUpdate(documentId=document_id, body={'requests': requests}).execute()


def move_document_to_folder(service_drive, document_id, folder_id):
    file = service_drive.files().get(fileId=document_id, fields='parents').execute()
    parents = file.get('parents')
    if parents:
        previous_parents = ",".join(parents)
    else:
        # Handle the case where parents is None or not iterable
        # For example, you might set previous_parents to an empty string
        previous_parents = ""
    file = service_drive.files().update(fileId=document_id,
                                        addParents=folder_id,
                                        removeParents=previous_parents,
                                        fields='id, parents').execute()

def create_folder(service_drive, folder_name, parent_id=None):
    query = f"name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder'"
    if parent_id:
        query += f" and '{parent_id}' in parents"

    response = service_drive.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    files = response.get('files', [])

    if files:
        return files[0].get('id')

    folder_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder'
    }
    if parent_id:
        folder_metadata['parents'] = [parent_id]

    folder = service_drive.files().create(body=folder_metadata, fields='id').execute()
    return folder.get('id')


def main(args):
    # File paths
    json_file_path = f"test_errors_all_cleansing.json"
    credentials_file_path = 'other_data/error-analysis-411406-f9ed98ee2df5_v2.json'

    # Setup
    SCOPES = ['https://www.googleapis.com/auth/documents', 'https://www.googleapis.com/auth/drive']
    data = load_json_data(json_file_path)
    service_docs = setup_google_docs_api(credentials_file_path, SCOPES)
    service_drive = build('drive', 'v3', credentials=service_docs._http.credentials)

    document_id = create_document(service_drive, service_docs, "all cleansing validation")

    folder_id = args.folder_id
    image_folder_id = create_folder(service_drive, "Images", folder_id)
    move_document_to_folder(service_drive, document_id, folder_id)

    # Insert data into the document
    num = 0
    # sorted_keys = sorted(data, key=lambda x: int(x.split("episode")[1]))


    for id, sample in tqdm(data.items(), total=len(data)):
        if sample["label"] == "Success":
            continue
        insert_text(service_docs, document_id, f"{id} ({num})" + '\n',indent_amount=1)
        for key, value in sample.items():
            if isinstance(value, str) and os.path.exists(value):
                img_id = upload_image_to_drive(service_drive, value, image_folder_id)
                try:
                    insert_image(service_docs, document_id, img_id)
                except:
                    print(f"Error in inserting image for {id} and {key}")
            else:
                content = f"{key}: {value}"
                insert_text(service_docs, document_id, content,indent_amount=1)
        insert_text(service_docs, document_id, "\n", indent_amount=1)
        time.sleep(5)
        num += 1

    move_document_to_folder(service_drive, document_id, folder_id)

    document_url = f"https://docs.google.com/document/d/{document_id}/edit"
    print(f"Document created. Document URL: {document_url}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_id", type=str, required=True)
    args = parser.parse_args()
    main(args)
