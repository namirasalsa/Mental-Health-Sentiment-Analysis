import streamlit as st
import pandas as pd
from PIL import Image
import plotly.express as px
# Dataset Overview
# Load the dataset
file_path = 'dataset/mental_health_sentiment.csv'
mental_health_data = pd.read_csv(file_path)


# Display dataset information
st.header('Dataset Information')
st.write(f"**Total Entries:** {mental_health_data.shape[0]}")
st.write(f"**Total Columns:** {mental_health_data.shape[1]}")

# Description of labels
st.subheader('Label Descriptions')
st.write("""
- **0 = very normal**: Indicates statements that are considered very normal and do not reflect any mental health concerns.
- **1 = normal**: Indicates statements that are normal and typical, with no apparent mental health issues.
- **2 = less normal**: Indicates statements that reflect some signs of mental health issues, less normal than typical.
- **3 = bad**: Indicates statements that reflect serious mental health concerns or distress.
""")

# Display first few rows of the dataset
st.subheader('Dataset Preview')
st.write(mental_health_data.head())

# Display full dataset in an expandable section
st.subheader('Full Dataset')
with st.expander('Show Full Dataset'):
    st.write(mental_health_data)

#EDA
st.header('Exploratory Data Analysis (EDA)')
#Class Distribution
# Load photo of distribution
st.header('Distribution of Labels')
# Menggabungkan data sebelum dan sesudah oversample
df_before = pd.DataFrame({'label': [0]*697 + [1]*16102 + [2]*10211 + [3]*967, 'Data': 'Before Oversample'})
df_after = pd.DataFrame({'label': [0]*16102 + [1]*16102 + [2]*10211 + [3]*16102, 'Data': 'After Oversample'})
df_before_after = pd.concat([df_before, df_after])

# Membuat bar plot menggunakan Plotly
fig = px.histogram(df_before_after, x='label', color='Data', barmode='group',
                   category_orders={'label': [0, 1, 2, 3]},
                   labels={'label': 'Label', 'count': 'Count'},
                   title='Distribution of Classes Before and After Oversample')

# Menampilkan plot di Streamlit
st.plotly_chart(fig)

# Analysis text
st.write("""
### Distribution Before Oversampling:
- Class 0: Very few samples, almost invisible on the graph.
- Class 1: Very large number of samples, around 16,000 samples.
- Class 2: Medium number of samples, around 11,000 samples.
- Class 3: Very few samples, almost invisible on the graph.

### Distribution After Oversampling:
- Class 0: Number of samples increased drastically to be equal to other classes, around 16,000 samples.
- Class 1: Number of samples remains the same, around 16,000 samples.
- Class 2: Number of samples increased to around 16,000 samples.
- Class 3: Number of samples increased drastically to be equal to other classes, around 16,000 samples.

**Conclusion:** The oversampling process successfully balanced the class distribution. Classes that were initially very underrepresented (Class 0 and Class 3) now have a number of samples equal to the other classes (Class 1 and Class 2). Thus, the class distribution after oversampling becomes more balanced, which is expected to improve the performance of the machine learning model in predicting all classes more fairly.
""")

#Word Cloud Analysis
# Load word cloud images
label_0_img = Image.open("asset/wc0.png")
label_1_img = Image.open("asset/wc1.png")
label_2_img = Image.open("asset/wc2.png")
label_3_img = Image.open("asset/wc3.png")
all_data_img = Image.open("asset/wc all.png")

# Streamlit application
st.header('Word Cloud Analysis Based on Labels')

st.subheader('Word Cloud for Label 0')
st.image(label_0_img, caption='Word Cloud for Label 0')
st.write('''
**Dominant Words:** "filler," "beautiful," "good day," "happy birthday," "everyone."

**Analysis:** The word cloud for label 0 shows many positive and commonly used social context words such as "beautiful," "good day," and "happy birthday." The word "filler" appears very dominant, possibly indicating a placeholder or filler word in the text.
''')

st.subheader('Word Cloud for Label 1')
st.image(label_1_img, caption='Word Cloud for Label 1')
st.write('''
**Dominant Words:** "one," "life," "know," "feel," "time."

**Analysis:** For label 1, the dominant words focus more on self-reflection and life experiences such as "life," "feel," and "know." Words like "one" and "time" also appear frequently, indicating a lot of discussions about time and individuals.
''')

st.subheader('Word Cloud for Label 2')
st.image(label_2_img, caption='Word Cloud for Label 2')
st.write('''
**Dominant Words:** "know," "life," "feel," "one," "want."

**Analysis:** Label 2 shares some key words with label 1 but adds words like "want," indicating desire or need. These words suggest a focus on personal experiences and feelings.
''')

st.subheader('Word Cloud for Label 3')
st.image(label_3_img, caption='Word Cloud for Label 3')
st.write('''
**Dominant Words:** "pain," "sad," "really," "filler," "done."

**Analysis:** The word cloud for label 3 highlights many words related to negative feelings such as "pain," "sad," and "done." The word "filler" remains dominant but contrasts with other emotional negative words, indicating content that may express difficulty or despair.
''')

st.subheader('Word Cloud for Entire Data')
st.image(all_data_img, caption='Word Cloud for Entire Data')
st.write('''
**Dominant Words:** "filler," "life," "one," "feel," "know."

**Analysis:** The word cloud representing the entire dataset shows a combination of words that frequently appear in all labels. The word "filler" remains dominant, but words like "life," "one," "feel," and "know" indicate general themes focused on life, feelings, and self-reflection.
''')