import pandas as pd
from urllib.parse import unquote
import wikitoolkit as wt
import pickle
import sqlite3
import time
import datetime
import os
from urllib3.exceptions import MaxRetryError
import asyncio

async def collect_topics():

    my_agent = 'wikireddit <p.gildersleve@exeter.ac.uk>'

    ranges_df = pd.read_hdf('data/ranges_df.h5')

    with open('data/langpagemaps.pkl', 'rb') as f:
        pagemapsdict = pickle.load(f)

    # get unique lang and article titles

    lang_title_dict = ranges_df[['lang', 'title']].drop_duplicates().groupby('lang')['title'].apply(list).to_dict()

    maxgroupsize = 1000
    l_topics_df = []
    xx = None
    for lang, articles in lang_title_dict.items():

        wtsession = wt.WTSession(f'{lang}.wikipedia', user_agent=my_agent)

        grouptopics = {}
        groupsize = maxgroupsize
        counter = 0
        la = None
        datetimenow = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        while len(articles) > 0:
            print(f'{datetimenow}. {lang}: {len(articles)} articles remaining. {len(grouptopics)} articles processed.', end='\r')
            try:
                a_topics = wt.get_articles_topics_sync(wtsession, articles[:groupsize],
                                            lang=lang, tf_args={'threshold': 0},
                                            pagemaps=pagemapsdict[lang])
                grouptopics.update(a_topics)
                articles = articles[groupsize:]
                groupsize = min(int(round(groupsize * (2**0.6), 0)), maxgroupsize)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                if len(articles) == la:
                    counter += 1
                    time.sleep(10)
                    if counter > 10:
                        print(f'{lang}: {len(articles)} articles remaining. {len(grouptopics)} articles processed.')
                        print(e, 'emergency_save')
                        l_topics_df = pd.concat(l_topics_df, ignore_index=True)
                        l_topics_df.to_hdf('data/topics_df_exit.h5', key='df')
                        with open('data/grouptopics.pkl_exit', 'wb') as f:
                            pickle.dump(grouptopics, f)
                        raise e

                print(e, 'Reducing group size to', groupsize // 2)
                time.sleep(0.1)
                groupsize = groupsize // 2

        grouptopics = pd.concat({k: pd.Series(v) for k, v in grouptopics.items()}).reset_index().rename(
                        columns={'level_0': 'article', 'level_1': 'topic', 0: 'score'})
        grouptopics = grouptopics.pivot(index='article', columns='topic', values='score').reset_index().rename_axis(None, axis=1)
        grouptopics['lang'] = lang
        grouptopics = grouptopics.set_index(['lang', 'article'])
        l_topics_df.append(grouptopics)
        
        await wtsession.close()

    l_topics_df = pd.concat(l_topics_df, ignore_index=True)
    l_topics_df.to_hdf('data/topics_df.h5', key='df')




# --- Entry Point ---
if __name__ == "__main__":

    # Run the main async function
    asyncio.run(collect_topics())