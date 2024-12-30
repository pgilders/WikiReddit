import pandas as pd
import pickle

my_agent = 'wikireddit <p.gildersleve@exeter.ac.uk>'

article_dates_unique = pd.read_hdf('data/article_dates_unique.h5', 'df')

article_dates_unique = article_dates_unique.dropna(subset=['pageid'])
article_dates_unique = article_dates_unique[article_dates_unique['pageid'] != -1]
article_dates_unique['pageid'] = article_dates_unique['pageid'].astype(int)

langs = article_dates_unique['lang'].unique()
date_article_dict = {}
for lang in langs:
    lang_article_dates_unique = article_dates_unique[article_dates_unique['lang'] == lang]

    temp_dfs = []
    # Iterate through each row in the dataframe
    for index, row in lang_article_dates_unique.iterrows():
        if index%100000 == 0:
            print(index/len(lang_article_dates_unique), end="\r")
        # Generate date range ±10 days for each date
        date_range = pd.date_range(start=row['date'] - pd.Timedelta(days=10), 
                                end=row['date'] + pd.Timedelta(days=10))
        
        # Create a temporary dataframe for the date range
        temp_df = pd.DataFrame({'redirected_title': row['redirected_title'], 
                                'date': date_range})
        temp_dfs.append(temp_df)

    # Concatenate the temporary dataframe to the expanded dataframe
    expanded_df = pd.concat(temp_dfs, ignore_index=True)

    # Display the expanded dataframe
    expanded_df = expanded_df.drop_duplicates()

    # Sort the dataframe
    print('sorting')
    expanded_df = expanded_df.sort_values(by=['redirected_title', 'date'])

    expanded_df.to_hdf(f'data/expanded_df.h5', f'/{lang}', mode='w')

    # Group by article and find continuous ranges
    print('grouping')
    result = []
    for (lang, article), group in expanded_df.groupby(['redirected_title']):
        group = group.reset_index(drop=True)
        group['gap'] = (group['date'] - group['date'].shift()).dt.days.ne(1).cumsum()  # Identify gaps
        ranges = group.groupby('gap')['date'].agg(['min', 'max']).reset_index(drop=True)
        for _, row in ranges.iterrows():
            result.append({'redirected_title': article, 'start_date': row['min'], 'end_date': row['max']})

    # Convert the result to a dataframe
    range_df = pd.DataFrame(result)
    range_df.to_hdf(f'data/range_df.h5', f'/{lang}', mode='w')

    print('dicting')
    for _, row in range_df.iterrows():
        date_range = (row['start_date'], row['end_date'])
        if lang not in date_article_dict:
            date_article_dict[lang] = {}
        if date_range not in date_article_dict[l]:
            date_article_dict[lang][date_range] = []
        date_article_dict[lang][date_range].append(row['redirected_title'])

with open('data/date_article_dict.pkl', 'wb') as f:
    pickle.dump(date_article_dict, f)

