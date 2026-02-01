import os
import pickle
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaFileUpload
import googleapiclient.discovery
import googleapiclient.errors
from dotenv import load_dotenv

# Load biến môi trường từ tệp .env
load_dotenv()

# Thông tin xác thực
API_KEY = os.getenv("YOUTUBE_DATA_API_KEY_V3")
CLIENT_SECRETS_FILE = "D:\\project\\Demo App\\OAuth 2.0 Client IDs\\client_secret_74133536206-9ho6diqn4kvu3siqo08iqrt2glvpne7c.apps.googleusercontent.com.json"
SCOPES = ['https://www.googleapis.com/auth/youtube.force-ssl']
CHANNEL_ID = os.getenv("YOUTUBE_CHANNEL_ID", "UC_x5XG1OV2P6uZZ5FSM9Ttw")
VIDEO_ID = os.getenv("YOUTUBE_VIDEO_ID", "G8EAuxNQaFM")
PLAYLIST_ID = os.getenv("YOUTUBE_PLAYLIST_ID", "PL9TGuasuEvY9YnueXPgMSog9gFfB66g9L")
COMMENT_ID = os.getenv("YOUTUBE_COMMENT_ID", "UgxZHkXVXHlzxdKpsMN4AaABAg")
CAPTION_ID = os.getenv("YOUTUBE_CAPTION_ID", "AUieDaYgLwGIBUkn-MqCvrWqTrvFMSQD-sg-kMoLjuGE6L3sKkg")
CHANNEL_SECTION_ID = os.getenv("YOUTUBE_CHANNEL_SECTION_ID", "UC_x5XG1OV2P6uZZ5FSM9Ttw.jNQXAC9IVRw")
SUBSCRIPTION_ID = os.getenv("YOUTUBE_SUBSCRIPTION_ID", "Cv7etA9TSgUlKfud0zE3QT95xqHK-owZN8og0dmI7Ac")

# Hàm lấy thông tin xác thực OAuth 2.0 với lưu trữ
def get_credentials():
    creds = None
    token_file = "token.pickle"
    if os.path.exists(token_file):
        with open(token_file, "rb") as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                print(f"Lỗi khi làm mới thông tin xác thực: {e}")
                creds = None
        if not creds:
            try:
                flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
                creds = flow.run_local_server(port=0)
                with open(token_file, "wb") as token:
                    pickle.dump(creds, token)
            except Exception as e:
                print(f"Lỗi khi xác thực OAuth: {e}")
                return None
    return creds

# Khởi tạo YouTube API client với API key
def get_youtube_client_api_key():
    try:
        return googleapiclient.discovery.build('youtube', 'v3', developerKey=API_KEY)
    except Exception as e:
        print(f"Lỗi khi khởi tạo client với API key: {e}")
        return None

# Khởi tạo YouTube API client với OAuth 2.0
def get_youtube_client_oauth():
    creds = get_credentials()
    if not creds:
        print("Không thể lấy thông tin xác thực OAuth.")
        return None
    try:
        return googleapiclient.discovery.build('youtube', 'v3', credentials=creds)
    except Exception as e:
        print(f"Lỗi khi khởi tạo client với OAuth: {e}")
        return None

# 1. Activities
def list_activities(youtube):
    try:
        request = youtube.activities().list(
            part='snippet,contentDetails',
            channelId=CHANNEL_ID,
            maxResults=5
        )
        response = request.execute()
        print("Activities:")
        if not response.get("items", []):
            print("Không có hoạt động nào.")
        else:
            for item in response['items']:
                print(f"Hoạt động: {item['snippet']['title']}, Type: {item['snippet']['type']}")
    except googleapiclient.errors.HttpError as e:
        print(f"Lỗi (Activities.list): {e}")

# 2. Captions
def list_captions(youtube):
    try:
        request = youtube.captions().list(
            part='snippet',
            videoId=VIDEO_ID
        )
        response = request.execute()
        print("Captions:")
        if not response.get("items", []):
            print(f"Không có phụ đề cho video ID: {VIDEO_ID}")
        else:
            for item in response['items']:
                print(f"Phụ đề: {item['snippet']['name']}, Ngôn ngữ: {item['snippet']['language']}")
    except googleapiclient.errors.HttpError as e:
        print(f"Lỗi (Captions.list): {e}")

def insert_caption(youtube):
    try:
        media = MediaFileUpload(r"D:\project\Demo App\data\caption_demo.srt", mimetype="application/x-subrip")
        request = youtube.captions().insert(
            part='snippet',
            body={
                'snippet': {
                    'videoId': "G8EAuxNQaFM",
                    'language': 'vi',
                    'name': 'Demo Caption',
                    'isDraft': True
                }
            },
            media_body=media
        )
        response = request.execute()
        print(f"Đã tải lên phụ đề: {response['snippet']['name']}")
    except googleapiclient.errors.HttpError as e:
        print(f"Lỗi (Captions.insert): {e}")

def update_caption(youtube):
    try:
        request = youtube.captions().update(
            part='snippet',
            body={
                'id': CAPTION_ID,
                'snippet': {
                    'videoId': VIDEO_ID,
                    'language': 'en',
                    'name': 'Updated Caption',
                    'isDraft': False
                }
            }
        )
        response = request.execute()
        print(f"Đã cập nhật phụ đề: {response['snippet']['name']}")
    except googleapiclient.errors.HttpError as e:
        print(f"Lỗi (Captions.update): {e}")

def download_caption(youtube):
    try:
        request = youtube.captions().download(
            id=CAPTION_ID
        )
        response = request.execute()
        print("Đã tải phụ đề")
    except googleapiclient.errors.HttpError as e:
        print(f"Lỗi (Captions.download): {e}")

def delete_caption(youtube):
    try:
        request = youtube.captions().delete(
            id=CAPTION_ID
        )
        request.execute()
        print("Đã xóa phụ đề")
    except googleapiclient.errors.HttpError as e:
        print(f"Lỗi (Captions.delete): {e}")

# 3. ChannelBanners
def insert_channel_banner(youtube):
    try:
        request = youtube.channelBanners().insert(
            media_body=r'D:\project\Demo App\data\banner.jpg'  # Thay bằng đường dẫn file ảnh (2560x1440px)
        )
        response = request.execute()
        print(f"Banner URL: {response['url']}")
    except googleapiclient.errors.HttpError as e:
        print(f"Lỗi (ChannelBanners.insert): {e}")

# 4. ChannelSections
def list_channel_sections(youtube):
    try:
        request = youtube.channelSections().list(
            part='snippet,contentDetails',
            channelId=CHANNEL_ID
        )
        response = request.execute()
        print("Channel Sections:")
        if not response.get("items", []):
            print(f"Không có mục kênh nào cho kênh ID: {CHANNEL_ID}")
        else:
            for item in response['items']:
                title = item['snippet'].get('title', '(no title)')
                section_type = item['snippet'].get('type', '(no type)')
                print(f"Section: {title}, Type: {section_type}")
    except googleapiclient.errors.HttpError as e:
        print(f"Lỗi (ChannelSections.list): {e}")

def insert_channel_section(youtube):
    try:
        request = youtube.channelSections().insert(
            part='snippet,contentDetails',
            body={
                'snippet': {
                    'type': 'singlePlaylist',
                    'title': 'Test Section',
                    'position': 0
                },
                'contentDetails': {
                    'playlists': [PLAYLIST_ID]
                }
            }
        )
        response = request.execute()
        print(f"Đã tạo section: {response['snippet']['title']}")
    except googleapiclient.errors.HttpError as e:
        print(f"Lỗi (ChannelSections.insert): {e}")

def update_channel_section(youtube):
    try:
        request = youtube.channelSections().update(
            part='snippet,contentDetails',
            body={
                'id': CHANNEL_SECTION_ID,
                'snippet': {
                    'type': 'singlePlaylist',
                    'title': 'Updated Section',
                    'position': 0
                },
                'contentDetails': {
                    'playlists': [PLAYLIST_ID]
                }
            }
        )
        response = request.execute()
        print(f"Đã cập nhật section: {response['snippet']['title']}")
    except googleapiclient.errors.HttpError as e:
        print(f"Lỗi (ChannelSections.update): {e}")

def delete_channel_section(youtube):
    try:
        request = youtube.channelSections().delete(
            id=CHANNEL_SECTION_ID
        )
        request.execute()
        print("Đã xóa section")
    except googleapiclient.errors.HttpError as e:
        print(f"Lỗi (ChannelSections.delete): {e}")

# 5. Channels
def list_channels(youtube):
    try:
        request = youtube.channels().list(
            part='snippet,statistics,brandingSettings',
            id=CHANNEL_ID
        )
        response = request.execute()
        print("Channels:")
        if not response.get("items", []):
            print("Không tìm thấy kênh.")
        else:
            for item in response['items']:
                print(f"Kênh: {item['snippet']['title']}, Subscribers: {item['statistics'].get('subscriberCount', 'N/A')}")
    except googleapiclient.errors.HttpError as e:
        print(f"Lỗi (Channels.list): {e}")

def update_channel(youtube):
    try:
        request = youtube.channels().update(
            part='brandingSettings',
            body={
                'id': CHANNEL_ID,
                'brandingSettings': {
                    'channel': {
                        'description': 'Updated channel description via API'
                    }
                }
            }
        )
        response = request.execute()
        print(f"Đã cập nhật kênh: {response['snippet']['title']}")
    except googleapiclient.errors.HttpError as e:
        print(f"Lỗi (Channels.update): {e}")

# 6. CommentThreads
def list_comment_threads(youtube):
    try:
        request = youtube.commentThreads().list(
            part='snippet',
            videoId=VIDEO_ID,
            maxResults=5
        )
        response = request.execute()
        print("Comment Threads:")
        if not response.get("items", []):
            print(f"Không có bình luận cho video ID: {VIDEO_ID}")
        else:
            for item in response['items']:
                print(f"Bình luận: {item['snippet']['topLevelComment']['snippet']['textOriginal']}")
    except googleapiclient.errors.HttpError as e:
        print(f"Lỗi (CommentThreads.list): {e}")

def insert_comment_thread(youtube):
    try:
        request = youtube.commentThreads().insert(
            part='snippet',
            body={
                'snippet': {
                    'videoId': VIDEO_ID,
                    'topLevelComment': {
                        'snippet': {
                            'textOriginal': 'Bình luận test từ API!'
                        }
                    }
                }
            }
        )
        response = request.execute()
        print(f"Đã thêm bình luận: {response['snippet']['topLevelComment']['snippet']['textOriginal']}")
    except googleapiclient.errors.HttpError as e:
        print(f"Lỗi (CommentThreads.insert): {e}")

# 7. Comments
def list_comments(youtube):
    try:
        request = youtube.comments().list(
            part='snippet',
            parentId=COMMENT_ID,
            maxResults=5
        )
        response = request.execute()
        print("Comments:")
        if not response.get("items", []):
            print(f"Không có bình luận con cho comment ID: {COMMENT_ID}")
        else:
            for item in response['items']:
                print(f"Bình luận: {item['snippet']['textOriginal']}")
    except googleapiclient.errors.HttpError as e:
        print(f"Lỗi (Comments.list): {e}")

def insert_comment(youtube):
    try:
        request = youtube.comments().insert(
            part='snippet',
            body={
                'snippet': {
                    'parentId': COMMENT_ID,
                    'textOriginal': 'Trả lời bình luận từ API!'
                }
            }
        )
        response = request.execute()
        print(f"Đã thêm trả lời: {response['snippet']['textOriginal']}")
    except googleapiclient.errors.HttpError as e:
        print(f"Lỗi (Comments.insert): {e}")

def update_comment(youtube):
    try:
        request = youtube.comments().update(
            part='snippet',
            body={
                'id': COMMENT_ID,
                'snippet': {
                    'textOriginal': 'Bình luận đã được chỉnh sửa!'
                }
            }
        )
        response = request.execute()
        print(f"Đã cập nhật bình luận: {response['snippet']['textOriginal']}")
    except googleapiclient.errors.HttpError as e:
        print(f"Lỗi (Comments.update): {e}")

def set_moderation_status(youtube):
    try:
        request = youtube.comments().setModerationStatus(
            id=COMMENT_ID,
            moderationStatus='published'
        )
        request.execute()
        print("Đã đặt trạng thái bình luận: published")
    except googleapiclient.errors.HttpError as e:
        print(f"Lỗi (Comments.setModerationStatus): {e}")

def delete_comment(youtube):
    try:
        request = youtube.comments().delete(
            id=COMMENT_ID
        )
        request.execute()
        print("Đã xóa bình luận")
    except googleapiclient.errors.HttpError as e:
        print(f"Lỗi (Comments.delete): {e}")

# 8. GuideCategories
def list_guide_categories(youtube):
    try:
        request = youtube.guideCategories().list(
            part='snippet',
            regionCode='US'
        )
        response = request.execute()
        print("Guide Categories:")
        if not response.get("items", []):
            print("Không có danh mục hướng dẫn.")
        else:
            for item in response['items']:
                print(f"Danh mục: {item['snippet']['title']}")
    except googleapiclient.errors.HttpError as e:
        print(f"Lỗi (GuideCategories.list): {e}")

# 8. GuideCategories (API này đã bị ngừng hỗ trợ, không sử dụng nữa)
# def list_guide_categories(youtube):
#     youtube = get_youtube_client_api_key()
#     try:
#         request = youtube.guideCategories().list(
#             part='snippet',
#             regionCode='US'
#         )
#         response = request.execute()
#         print("Guide Categories:")
#         if not response.get("items", []):
#             print("Không có danh mục hướng dẫn.")
#         else:
#             for item in response['items']:
#                 print(f"Danh mục: {item['snippet']['title']}")
#     except googleapiclient.errors.HttpError as e:
#         print(f"Lỗi (GuideCategories.list): {e}")

# 9. I18nLanguages
def list_i18n_languages(youtube):
    try:
        request = youtube.i18nLanguages().list(
            part='snippet'
        )
        response = request.execute()
        print("I18n Languages:")
        if not response.get("items", []):
            print("Không có ngôn ngữ i18n.")
        else:
            for item in response['items']:
                print(f"Ngôn ngữ: {item['snippet']['name']}, Code: {item['snippet']['hl']}")
    except googleapiclient.errors.HttpError as e:
        print(f"Lỗi (I18nLanguages.list): {e}")

# 10. I18nRegions
def list_i18n_regions(youtube):
    try:
        request = youtube.i18nRegions().list(
            part='snippet'
        )
        response = request.execute()
        print("I18n Regions:")
        if not response.get("items", []):
            print("Không có khu vực i18n.")
        else:
            for item in response['items']:
                print(f"Khu vực: {item['snippet']['name']}, Code: {item['snippet']['gl']}")
    except googleapiclient.errors.HttpError as e:
        print(f"Lỗi (I18nRegions.list): {e}")

# 11. Members
def list_members(youtube):
    try:
        request = youtube.members().list(
            part='snippet',
            maxResults=5
        )
        response = request.execute()
        print("Members:")
        if not response.get("items", []):
            print("Không có thành viên.")
        else:
            for item in response['items']:
                print(f"Thành viên: {item['snippet']['memberDetails']['channelId']}")
    except googleapiclient.errors.HttpError as e:
        print(f"Lỗi (Members.list): {e}")

# 12. MembershipsLevels
def list_membership_levels(youtube):
    try:
        request = youtube.membershipsLevels().list(
            part='snippet'
        )
        response = request.execute()
        print("Membership Levels:")
        if not response.get("items", []):
            print("Không có cấp độ thành viên.")
        else:
            for item in response['items']:
                print(f"Cấp độ: {item['snippet']['title']}")
    except googleapiclient.errors.HttpError as e:
        print(f"Lỗi (MembershipsLevels.list): {e}")

# 13. PlaylistItems
def list_playlist_items(youtube):
    try:
        request = youtube.playlistItems().list(
            part='snippet',
            playlistId=PLAYLIST_ID,
            maxResults=5
        )
        response = request.execute()
        print("Playlist Items:")
        if not response.get("items", []):
            print(f"Không có item trong playlist ID: {PLAYLIST_ID}")
        else:
            for item in response['items']:
                print(f"Video: {item['snippet']['title']}")
    except googleapiclient.errors.HttpError as e:
        print(f"Lỗi (PlaylistItems.list): {e}")

def insert_playlist_item(youtube):
    try:
        request = youtube.playlistItems().insert(
            part='snippet',
            body={
                'snippet': {
                    'playlistId': PLAYLIST_ID,
                    'resourceId': {
                        'kind': 'youtube#video',
                        'videoId': VIDEO_ID
                    }
                }
            }
        )
        response = request.execute()
        print(f"Đã thêm video vào playlist: {response['snippet']['title']}")
    except googleapiclient.errors.HttpError as e:
        print(f"Lỗi (PlaylistItems.insert): {e}")

def update_playlist_item(youtube):
    try:
        request = youtube.playlistItems().update(
            part='snippet',
            body={
                'id': 'YOUR_PLAYLIST_ITEM_ID',  # Thay bằng playlist item ID thực tế
                'snippet': {
                    'playlistId': PLAYLIST_ID,
                    'position': 1,
                    'resourceId': {
                        'kind': 'youtube#video',
                        'videoId': VIDEO_ID
                    }
                }
            }
        )
        response = request.execute()
        print(f"Đã cập nhật playlist item: {response['snippet']['title']}")
    except googleapiclient.errors.HttpError as e:
        print(f"Lỗi (PlaylistItems.update): {e}")

def delete_playlist_item(youtube):
    try:
        request = youtube.playlistItems().delete(
            id='YOUR_PLAYLIST_ITEM_ID'  # Thay bằng playlist item ID thực tế
        )
        request.execute()
        print("Đã xóa playlist item")
    except googleapiclient.errors.HttpError as e:
        print(f"Lỗi (PlaylistItems.delete): {e}")

# 14. Playlists
def list_playlists(youtube):
    try:
        request = youtube.playlists().list(
            part='snippet,contentDetails',
            channelId=CHANNEL_ID,
            maxResults=5
        )
        response = request.execute()
        print("Playlists:")
        if not response.get("items", []):
            print(f"Không có playlist cho kênh ID: {CHANNEL_ID}")
        else:
            for item in response['items']:
                print(f"Playlist: {item['snippet']['title']}, ID: {item['id']}")
    except googleapiclient.errors.HttpError as e:
        print(f"Lỗi (Playlists.list): {e}")

def insert_playlist(youtube):
    try:
        request = youtube.playlists().insert(
            part='snippet,status',
            body={
                'snippet': {
                    'title': 'Test Playlist',
                    'description': 'Playlist created via API'
                },
                'status': {
                    'privacyStatus': 'public'
                }
            }
        )
        response = request.execute()
        print(f"Đã tạo playlist: {response['snippet']['title']}, ID: {response['id']}")
    except googleapiclient.errors.HttpError as e:
        print(f"Lỗi (Playlists.insert): {e}")

def update_playlist(youtube):
    try:
        request = youtube.playlists().update(
            part='snippet',
            body={
                'id': PLAYLIST_ID,
                'snippet': {
                    'title': 'Updated Playlist',
                    'description': 'Updated description via API'
                }
            }
        )
        response = request.execute()
        print(f"Đã cập nhật playlist: {response['snippet']['title']}")
    except googleapiclient.errors.HttpError as e:
        print(f"Lỗi (Playlists.update): {e}")

def delete_playlist(youtube):
    try:
        request = youtube.playlists().delete(
            id=PLAYLIST_ID
        )
        request.execute()
        print("Đã xóa playlist")
    except googleapiclient.errors.HttpError as e:
        print(f"Lỗi (Playlists.delete): {e}")

# 15. Search
def search_videos(youtube):
    try:
        request = youtube.search().list(
            part='snippet',
            q='python tutorial',
            type='video',
            maxResults=5
        )
        response = request.execute()
        print("Search Results:")
        if not response.get("items", []):
            print("Không có kết quả tìm kiếm.")
        else:
            for item in response['items']:
                print(f"Video: {item['snippet']['title']}, Video ID: {item['id']['videoId']}")
    except googleapiclient.errors.HttpError as e:
        print(f"Lỗi (Search.list): {e}")

# 16. Subscriptions
def list_subscriptions(youtube):
    try:
        request = youtube.subscriptions().list(
            part='snippet',
            mine=True,
            maxResults=5
        )
        response = request.execute()
        print("Subscriptions:")
        if not response.get("items", []):
            print("Không có đăng ký nào.")
        else:
            for item in response['items']:
                print(f"Kênh: {item['snippet']['title']}")
    except googleapiclient.errors.HttpError as e:
        print(f"Lỗi (Subscriptions.list): {e}")

def insert_subscription(youtube):
    try:
        request = youtube.subscriptions().insert(
            part='snippet',
            body={
                'snippet': {
                    'resourceId': {
                        'kind': 'youtube#channel',
                        'channelId': CHANNEL_ID
                    }
                }
            }
        )
        response = request.execute()
        print(f"Đã đăng ký kênh: {response['snippet']['title']}")
    except googleapiclient.errors.HttpError as e:
        print(f"Lỗi (Subscriptions.insert): {e}")

def delete_subscription(youtube):
    try:
        request = youtube.subscriptions().delete(
            id=SUBSCRIPTION_ID
        )
        request.execute()
        print("Đã hủy đăng ký")
    except googleapiclient.errors.HttpError as e:
        print(f"Lỗi (Subscriptions.delete): {e}")

# 17. Thumbnails
def set_thumbnail(youtube):
    try:
        request = youtube.thumbnails().set(
            videoId=VIDEO_ID,
            media_body=r'D:\project\Demo App\data\thumbnail.jpg'  # Thay bằng đường dẫn file ảnh
        )
        response = request.execute()
        print("Đã tải lên thumbnail")
    except googleapiclient.errors.HttpError as e:
        print(f"Lỗi (Thumbnails.set): {e}")

# 18. VideoAbuseReportReasons
def list_video_abuse_report_reasons(youtube):
    try:
        request = youtube.videoAbuseReportReasons().list(
            part='snippet'
        )
        response = request.execute()
        print("Video Abuse Report Reasons:")
        if not response.get("items", []):
            print("Không có lý do báo cáo.")
        else:
            for item in response['items']:
                print(f"Lý do: {item['snippet']['label']}")
    except googleapiclient.errors.HttpError as e:
        print(f"Lỗi (VideoAbuseReportReasons.list): {e}")

# 19. VideoCategories
def list_video_categories(youtube):
    try:
        request = youtube.videoCategories().list(
            part='snippet',
            regionCode='US'
        )
        response = request.execute()
        print("Video Categories:")
        if not response.get("items", []):
            print("Không có danh mục video.")
        else:
            for item in response['items']:
                print(f"Danh mục: {item['snippet']['title']}")
    except googleapiclient.errors.HttpError as e:
        print(f"Lỗi (VideoCategories.list): {e}")

# 20. Videos
def list_videos(youtube):
    try:
        request = youtube.videos().list(
            part='snippet,statistics',
            id=VIDEO_ID
        )
        response = request.execute()
        print("Videos:")
        if not response.get("items", []):
            print(f"Không có video với ID: {VIDEO_ID}")
        else:
            for item in response['items']:
                print(f"Video: {item['snippet']['title']}, Views: {item['statistics'].get('viewCount', 'N/A')}")
    except googleapiclient.errors.HttpError as e:
        print(f"Lỗi (Videos.list): {e}")

def insert_video(youtube):
    try:
        request = youtube.videos().insert(
            part='snippet,status',
            body={
                'snippet': {
                    'title': 'Test Video',
                    'description': 'Video uploaded via API',
                    'categoryId': '22'  # Danh mục "People & Blogs"
                },
                'status': {
                    'privacyStatus': 'public'
                }
            },
            media_body=r'D:\project\Demo App\data\video.mp4'  # Thay bằng đường dẫn file video
        )
        response = request.execute()
        print(f"Đã tải lên video: {response['snippet']['title']}")
    except googleapiclient.errors.HttpError as e:
        print(f"Lỗi (Videos.insert): {e}")

def update_video(youtube):
    try:
        request = youtube.videos().update(
            part='snippet',
            body={
                'id': VIDEO_ID,
                'snippet': {
                    'title': 'Updated Video Title',
                    'description': 'Updated description via API',
                    'categoryId': '22'
                }
            }
        )
        response = request.execute()
        print(f"Đã cập nhật video: {response['snippet']['title']}")
    except googleapiclient.errors.HttpError as e:
        print(f"Lỗi (Videos.update): {e}")

def delete_video(youtube):
    try:
        request = youtube.videos().delete(
            id=VIDEO_ID
        )
        request.execute()
        print("Đã xóa video")
    except googleapiclient.errors.HttpError as e:
        print(f"Lỗi (Videos.delete): {e}")

def rate_video(youtube):
    try:
        request = youtube.videos().rate(
            id=VIDEO_ID,
            rating='like'
        )
        request.execute()
        print("Đã thích video")
    except googleapiclient.errors.HttpError as e:
        print(f"Lỗi (Videos.rate): {e}")

def get_video_rating(youtube):
    try:
        request = youtube.videos().getRating(
            id=VIDEO_ID
        )
        response = request.execute()
        if not response.get("items", []):
            print("Không có đánh giá.")
        else:
            print(f"Đánh giá video: {response['items'][0]['rating']}")
    except googleapiclient.errors.HttpError as e:
        print(f"Lỗi (Videos.getRating): {e}")

def report_abuse_video(youtube):
    try:
        request = youtube.videos().reportAbuse(
            body={
                'videoId': VIDEO_ID,
                'reasonId': 'YOUR_REASON_ID'  # Thay bằng reason ID từ videoAbuseReportReasons
            }
        )
        request.execute()
        print("Đã báo cáo video")
    except googleapiclient.errors.HttpError as e:
        print(f"Lỗi (Videos.reportAbuse): {e}")

# 21. Watermarks
def set_watermark(youtube):
    try:
        request = youtube.watermarks().set(
            channelId=CHANNEL_ID,
            body={
                'timing': {
                    'offsetMs': 0,
                    'durationMs': 10000
                }
            },
            media_body=r'D:\project\Demo App\data\watermark.png'  # Thay bằng đường dẫn file ảnh
        )
        response = request.execute()
        print("Đã thiết lập watermark")
    except googleapiclient.errors.HttpError as e:
        print(f"Lỗi (Watermarks.set): {e}")

def unset_watermark(youtube):
    try:
        request = youtube.watermarks().unset(
            channelId=CHANNEL_ID
        )
        request.execute()
        print("Đã xóa watermark")
    except googleapiclient.errors.HttpError as e:
        print(f"Lỗi (Watermarks.unset): {e}")

# Chạy các hàm demo
if __name__ == '__main__':
    print("--- Khởi tạo Clients ---")
    youtube_api_key = get_youtube_client_api_key()
    youtube_oauth = get_youtube_client_oauth()
    
    if not youtube_api_key:
        print("Không thể khởi tạo client API key. Bỏ qua các hàm sử dụng API key.")
    
    if not youtube_oauth:
        print("Không thể khởi tạo client OAuth. Bỏ qua các hàm sử dụng OAuth.")
    
    if youtube_api_key:
        print("\n--- Activities ---")
        list_activities(youtube_api_key)
    
    if youtube_oauth:
        print("\n--- Captions ---")
        list_captions(youtube_oauth)
        # insert_caption(youtube_oauth)
        # update_caption(youtube_oauth)
        # download_caption(youtube_oauth)
        # delete_caption(youtube_oauth)
    
    if youtube_oauth:
        print("\n--- ChannelBanners ---")
        insert_channel_banner(youtube_oauth)
    
    if youtube_api_key:
        print("\n--- ChannelSections ---")
        list_channel_sections(youtube_api_key)
    
    # if youtube_oauth:
    #     print("\n--- ChannelSections OAuth ---")
        # insert_channel_section(youtube_oauth)
        # update_channel_section(youtube_oauth)
        # delete_channel_section(youtube_oauth)
    
    if youtube_api_key:
        print("\n--- Channels ---")
        list_channels(youtube_api_key)
    
    # if youtube_oauth:
    #     print("\n--- Channels OAuth ---")
    #     update_channel(youtube_oauth)
    
    if youtube_api_key:
        print("\n--- CommentThreads ---")
        list_comment_threads(youtube_api_key)
    
    # if youtube_oauth:
    #     print("\n--- CommentThreads OAuth ---")
    #     insert_comment_thread(youtube_oauth)
    
    if youtube_api_key:
        print("\n--- Comments ---")
        list_comments(youtube_api_key)
    
    # if youtube_oauth:
    #     print("\n--- Comments OAuth ---")
    #     insert_comment(youtube_oauth)
    #     update_comment(youtube_oauth)
    #     set_moderation_status(youtube_oauth)
    #     delete_comment(youtube_oauth)
    
    # if youtube_api_key:
    #     print("\n--- GuideCategories ---")
    #     list_guide_categories(youtube_api_key)
    
    if youtube_api_key:
        print("\n--- I18nLanguages ---")
        list_i18n_languages(youtube_api_key)
    
    if youtube_api_key:
        print("\n--- I18nRegions ---")
        list_i18n_regions(youtube_api_key)
    
    if youtube_oauth:
        print("\n--- Members ---")
        list_members(youtube_oauth)
    
    if youtube_oauth:
        print("\n--- MembershipsLevels ---")
        list_membership_levels(youtube_oauth)
    
    if youtube_api_key:
        print("\n--- PlaylistItems ---")
        list_playlist_items(youtube_api_key)
    
    # if youtube_oauth:
    #     print("\n--- PlaylistItems OAuth ---")
    #     insert_playlist_item(youtube_oauth)
    #     update_playlist_item(youtube_oauth)
    #     delete_playlist_item(youtube_oauth)
    
    if youtube_api_key:
        print("\n--- Playlists ---")
        list_playlists(youtube_api_key)
    
    # if youtube_oauth:
    #     print("\n--- Playlists OAuth ---")
    #     insert_playlist(youtube_oauth)
    #     update_playlist(youtube_oauth)
    #     delete_playlist(youtube_oauth)
    
    if youtube_api_key:
        print("\n--- Search ---")
        search_videos(youtube_api_key)
    
    if youtube_oauth:
        print("\n--- Subscriptions ---")
        list_subscriptions(youtube_oauth)
        # insert_subscription(youtube_oauth)
        # delete_subscription(youtube_oauth)
    
    # if youtube_oauth:
    #     print("\n--- Thumbnails ---")
    #     set_thumbnail(youtube_oauth)
    
    if youtube_api_key:
        print("\n--- VideoAbuseReportReasons ---")
        list_video_abuse_report_reasons(youtube_api_key)
    
    if youtube_api_key:
        print("\n--- VideoCategories ---")
        list_video_categories(youtube_api_key)
    
    if youtube_api_key:
        print("\n--- Videos ---")
        list_videos(youtube_api_key)
    
    # if youtube_oauth:
    #     print("\n--- Videos OAuth ---")
    #     insert_video(youtube_oauth)
    #     update_video(youtube_oauth)
    #     delete_video(youtube_oauth)
    #     rate_video(youtube_oauth)
    #     get_video_rating(youtube_oauth)
    #     report_abuse_video(youtube_oauth)
    
    # if youtube_oauth:
    #     print("\n--- Watermarks ---")
    #     set_watermark(youtube_oauth)
    #     unset_watermark(youtube_oauth)

    print("\n--- Hoàn thành demo ---")