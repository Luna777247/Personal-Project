import os
import pickle
import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
from google.auth.transport.requests import Request

CLIENT_SECRET_FILE = os.getenv("YOUTUBE_CLIENT_SECRET_FILE", r"D:\project\Demo App\OAuth 2.0 Client IDs\client_secret_74133536206-9ho6diqn4kvu3siqo08iqrt2glvpne7c.apps.googleusercontent.com.json")
SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]

def get_credentials():
    creds = None
    token_file = "token.pickle"
    if os.path.exists(token_file):
        with open(token_file, "rb") as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
            creds = flow.run_local_server(port=8080)
        with open(token_file, "wb") as token:
            pickle.dump(creds, token)
    return creds

def list_captions(youtube, video_id):
    print("\n=== Captions (Phụ đề) ===")
    resp = youtube.captions().list(
        part="snippet",
        videoId=video_id
    ).execute()
    for item in resp.get("items", []):
        print(f"Caption ID: {item['id']} | Name: {item['snippet'].get('name', '')} | Language: {item['snippet'].get('language', '')}")
        
def list_channel_sections(youtube, channel_id):
    print("\n=== Channel Sections ===")
    resp = youtube.channelSections().list(
        part="snippet,contentDetails",
        channelId=channel_id
    ).execute()
    for item in resp.get("items", []):
        print(f"Channel Section ID: {item['id']} | Title: {item['snippet'].get('title', '')} | Type: {item['snippet'].get('type', '')}")
   
   
def list_subscriptions(youtube):
    print("\n=== Subscriptions ===")
    resp = youtube.subscriptions().list(
        part="snippet",
        mine=True,
        maxResults=5
    ).execute()
    for item in resp.get("items", []):
        print(f"Subscription ID: {item['id']} | Channel Title: {item['snippet']['title']}")
             
def main():
    try:
        # Xác thực
        creds = get_credentials()
        youtube = googleapiclient.discovery.build("youtube", "v3", credentials=creds)

        # Lấy thông tin kênh
        print("\n=== Kênh YouTube của tôi ===")
        try:
            channel_resp = youtube.channels().list(part="snippet,statistics", mine=True).execute()
            if not channel_resp.get("items", []):
                print("Không tìm thấy thông tin kênh.")
            else:
                for item in channel_resp["items"]:
                    print(f"Tiêu đề     : {item['snippet']['title']}")
                    print(f"Mô tả       : {item['snippet']['description']}")
                    print(f"Người đăng ký: {item['statistics'].get('subscriberCount', 'N/A')}")
                    print(f"Lượt xem    : {item['statistics'].get('viewCount', 'N/A')}")
                    print(f"Video       : {item['statistics'].get('videoCount', 'N/A')}")
        except googleapiclient.errors.HttpError as e:
            print(f"Đã xảy ra lỗi khi lấy thông tin kênh: {e}")

        # Tìm kiếm video
        print("\n=== Tìm kiếm video (Python) ===")
        try:
            query = input("Nhập từ khóa tìm kiếm (mặc định: python): ") or "python"
            search_resp = youtube.search().list(q=query, part="snippet", type="video", maxResults=2).execute()
            if not search_resp.get("items", []):
                print(f"Không tìm thấy video nào với từ khóa '{query}'.")
            else:
                for item in search_resp["items"]:
                    print(f"Video: {item['snippet']['title']} (ID: {item['id']['videoId']})")
        except googleapiclient.errors.HttpError as e:
            print(f"Đã xảy ra lỗi khi tìm kiếm video: {e}")

        # Các chức năng khác tương tự...

    except Exception as e:
        print(f"Đã xảy ra lỗi chung: {e}")

if __name__ == "__main__":
    main()
    creds = get_credentials()
    youtube = googleapiclient.discovery.build("youtube", "v3", credentials=creds)
    video_id = "G8EAuxNQaFM"
    channel_id = "UC_x5XG1OV2P6uZZ5FSM9Ttw"
    list_captions(youtube, video_id)
    list_channel_sections(youtube, channel_id)
    list_subscriptions(youtube)