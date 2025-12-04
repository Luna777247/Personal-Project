# Data Sources for Football Image Captioning

## 1. Kaggle Datasets
- **Football Player Images**: https://www.kaggle.com/datasets/jorgebuenoperez/football-players
- **Soccer Ball Detection**: https://www.kaggle.com/datasets/soumikrakshit/soccer-ball-detection
- **Football Match Images**: https://www.kaggle.com/datasets/idoyoabraham/football-match-images
- **Premier League Images**: https://www.kaggle.com/datasets/thec03u5/premier-league-players

## 2. Academic Datasets
- **Sports Image Datasets**: University repositories
- **Flickr Sports Collection**: Filtered sports images
- **ImageNet Sports Categories**: Subset of ImageNet with sports images

## 3. Public APIs
- **Flickr API**: Search for football images with creative commons license
- **Unsplash API**: Sports photography
- **Pexels API**: Football and sports images

## 4. Web Scraping (with permission)
- **Football websites**: Official team websites, sports news sites
- **Social media**: Twitter, Instagram with public APIs
- **Sports archives**: Getty Images, Reuters sports photos

## 5. Generated/Synthetic Data
- **Sports highlight frames**: Extract from YouTube videos (fair use)
- **Augmented data**: Rotate, crop existing football images
- **Mixed datasets**: Combine multiple sources

## Recommended Approach:
1. Start with Kaggle football datasets
2. Supplement with Flickr API for more variety
3. Use data augmentation to increase dataset size
4. Ensure proper licensing and attribution

## Data Collection Script
See `src/data_collection.py` for automated data gathering