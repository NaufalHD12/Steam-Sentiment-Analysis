# Steam-Sentiment-Analysis
 
# Steam Game Reviews Sentiment Analysis

## Project Overview
This project performs sentiment analysis on a 10% sample of Steam game reviews using machine learning techniques. The goal is to understand user sentiments towards different games and develop a model that can automatically classify reviews as positive or negative.

## Dataset
- **Total Reviews**: 622,599 (10% sample)
- **Sentiment Distribution**:
  - Positive Reviews (Recommended): 513,031 (82.40%)
  - Negative Reviews (Not Recommended): 109,568 (17.60%)
- Link: https://www.kaggle.com/datasets/andrewmvd/steam-reviews/data

## Methodology

### Data Preprocessing
- Cleaned and processed review text data
- Handled missing values
- Retained 97.02% of the original sample

### Feature Extraction
- Used TF-IDF Vectorization
- Maximum of 5,000 features selected

### Machine Learning Model
- Algorithm: Multinomial Naive Bayes Classifier
- Model Performance:
  - Accuracy: 0.85
  - F1-Score: 0.92

## Key Insights

### Top Games by Positive Sentiment
1. Terraria (8,205 positive reviews)
2. Dota 2 (6,227 positive reviews)
3. Rust (6,150 positive reviews)
4. PAYDAY 2 (6,110 positive reviews)
5. DayZ (5,801 positive reviews)

### Top Games by Negative Sentiment
1. DayZ (3,007 negative reviews)
2. PAYDAY 2 (2,743 negative reviews)
3. Rust (1,573 negative reviews)
4. Robocraft (1,278 negative reviews)
5. Heroes & Generals (1,194 negative reviews)

### Most Informative Features for Positive Sentiment
- Words like 'underrated', 'fun', and 'awesome' strongly indicate positive reviews

## Limitations
- Analysis based on a 10% sample of the dataset
- Results may vary with different sampling
- Model performance might differ on the full dataset

## Tools and Libraries
- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- TensorFlow/Keras (optional for future work)

## How to Reproduce
1. Clone the repository
2. Install required dependencies
3. Run preprocessing scripts
4. Execute sentiment analysis notebook
5. Generate visualizations and analysis

**Note**: This analysis provides insights into Steam game reviews and demonstrates the potential of automated sentiment classification in understanding user experiences.
