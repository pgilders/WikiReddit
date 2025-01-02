import markdown
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urlunparse
import re
import pandas as pd
import requests
import asyncio
import aiohttp
from unicodedata import category
import time
import pickle
import os
import datetime

# Assume 'posts' DataFrame is already loaded, containing a 'body' column.
# posts = pd.read_hdf('data/posts.h5').reset_index(drop=True)

HEADERS = {'user-agent': 'wikireddit p.gildersleve@exeter.ac.uk'}
MAX_CONCURRENCY = 100

if os.path.exists('.temp/rd_url_cache_p.pkl'):
    with open('.temp/rd_url_cache_p.pkl', 'rb') as f:
        RD_URL_CACHE_P = pickle.load(f)
else:
    RD_URL_CACHE_P = {True: {}, False: {}}

if os.path.exists('.temp/rd_url_cache_tp.pkl'):
    with open('.temp/rd_url_cache_tp.pkl', 'rb') as f:
        RD_URL_CACHE_TP = pickle.load(f)
else:
    RD_URL_CACHE_TP = {}

import markdown
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urlunparse
import re
import pandas as pd
import requests
import asyncio
import aiohttp
from unicodedata import category
import time
import pickle
import os

HEADERS = {'user-agent': 'wikireddit p.gildersleve@exeter.ac.uk'}
MAX_CONCURRENCY = 100

if os.path.exists('.temp/rd_url_cache_p.pkl'):
    with open('.temp/rd_url_cache_p.pkl', 'rb') as f:
        RD_URL_CACHE_P = pickle.load(f)
else:
    RD_URL_CACHE_P = {True: {}, False: {}}
    
if os.path.exists('.temp/rd_url_cache_tp.pkl'):
    with open('.temp/rd_url_cache_tp.pkl', 'rb') as f:
        RD_URL_CACHE_TP = pickle.load(f)
else:
    RD_URL_CACHE_TP = {}

# --- Helper Functions ---

def clean_wikipedia_domain(link):
    if 'wikipedia.org' not in link:
        return None
    try:
        parsed = urlparse(link)
    except ValueError as ex:
        print(ex)
        return None
    domain = parsed.netloc.lower()
    while domain and category(domain[-1])[0]=='P':
        domain = domain[:-1]
    # get subdomain
    # subdomain = domain.split('wikipedia.org')[0]
    # if (subdomain not in subdomains) and subdomain:
    #     return None
    # Domain must be exactly 'wikipedia.org' or end with '.wikipedia.org'
    if domain == 'wikipedia.org' or domain.endswith('.wikipedia.org'):
        return urlunparse(('https', domain, parsed.path, parsed.params, parsed.query, parsed.fragment))
    else:
        # print(link, 'not wikipedia')
        return None

URL_REGEX = re.compile(
    r'(https?://[^\s<>\[\]{}|]+(?:wikipedia\.org)[^\s<>\[\]{}|]*)|((?:[a-z0-9-]+\.)*wikipedia\.org[^\s<>\[\]{}|]*)',
    re.IGNORECASE
)

def extract_plain_links(text):
    plain_links = []
    for match in URL_REGEX.finditer(text):
        candidate = match.group(1) if match.group(1) else match.group(2)
        plain_links.append(candidate.strip())
    return plain_links

def normalize_links(links):
    normalized = []
    for link in links:
        if not re.match(r'^https?://', link, re.IGNORECASE):
            link = 'https://' + link
        normalized.append(link)
    return normalized

def filter_wikipedia_links(links):
    wiki_links = set()
    for link in links:
        cleanlink = clean_wikipedia_domain(link)
        if cleanlink:
            wiki_links.add(cleanlink)
    return wiki_links

def extract_links_from_text(text):
    # Convert Markdown to HTML and extract markdown-parsed URLs
    html = markdown.markdown(text, extensions=['extra'])
    soup = BeautifulSoup(html, 'html.parser')
    a_tags = soup.find_all('a', href=True)
    markdown_links = [y.strip().replace(' ', '_') for a in a_tags for y in re.split(r'(?<=\s[^\w\s])|(?<=[^\w\s])\s', a['href'].strip())]
    markdown_links = [y for x in markdown_links for y in extract_plain_links(x)]

    # Remove these <a> tags to avoid double counting
    for a_tag in a_tags:
        a_tag.decompose()
    cleaned_text = soup.get_text()

    # Extract plain links
    plain_links = extract_plain_links(cleaned_text)

    # Combine, normalize, and deduplicate
    all_links = set(normalize_links(markdown_links + plain_links))

    # Filter to Wikipedia links
    wiki_links = filter_wikipedia_links(all_links)
    return list(wiki_links)

def reextract_links(text):
    plain_links = extract_plain_links(text)
    all_links = set(normalize_links(plain_links))
    wiki_links = filter_wikipedia_links(all_links)
    return list(wiki_links)

# --- Async Validation Functions ---
async def async_validate_link(session, url, timeout=5, retries=10, allow_redirects=False):
    try:
        async with session.head(url, allow_redirects=allow_redirects, timeout=timeout) as r:
            if 200 <= r.status < 400:
                return True, r.status, str(r.url)
            elif r.status == 429 and retries > 0:
                await asyncio.sleep(1)  # Wait for 1 second before retrying
                return await async_validate_link(session, url, timeout, retries - 1, allow_redirects=allow_redirects)
            else:
                async with session.get(url, allow_redirects=allow_redirects, timeout=timeout) as r2:
                    if 200 <= r2.status < 400:
                        return True, r2.status, str(r2.url)
                    else:
                        return False, r2.status, str(r2.url)
    except Exception as ex:
        print(ex)
    return False, -1, None

async def async_validate_url_with_punctuation(session, url, timeout=5, retries=10, allow_redirects=False):
    is_valid, status, processed_url = await async_validate_link(session, url, timeout, retries, allow_redirects)
    if is_valid:
        return is_valid, status, processed_url

    # Try removing trailing punctuation
    while url and category(url[-1])[0]=='P':
        url = url[:-1]
        is_valid, status, processed_url = await async_validate_link(session, url, timeout, retries, allow_redirects)
        if is_valid:
            return is_valid, status, processed_url

    return False, status, processed_url

async def async_validate_url_with_textpunctuation(session, url, timeout=5):
    is_valid, status, processed_url = await async_validate_link(session, url, timeout)
    if is_valid:
        return is_valid, status, processed_url

    # Remove trailing alphanumeric and punctuation until punctuation is hit
    original_processed_url = processed_url
    # print('av', url)
    url = url.rstrip('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789:\'"_')
    if url[-5:] != 'wiki/' and url[-5:] != '.org/':
        is_valid, status, processed_url = await async_validate_url_with_punctuation(session, url, timeout=5)
        if is_valid:
            return is_valid, status, processed_url

    return False, status, original_processed_url

async def validate_urls_main(urls, retries=10, allow_redirects=False, max_concurrency=MAX_CONCURRENCY):
    semaphore = asyncio.Semaphore(max_concurrency)
    async with aiohttp.ClientSession(headers=HEADERS) as session:
        async def validate(url):
            if url in RD_URL_CACHE_P[allow_redirects]:
                return url, RD_URL_CACHE_P[allow_redirects][url]  # Return cached result
            
            # ṬODO, add str vaidatopm here for domain

            async with semaphore:
                result = await async_validate_url_with_punctuation(session, url,
                                                                      retries=retries,
                                                                      allow_redirects=allow_redirects)
                if result[1] != 429:
                    RD_URL_CACHE_P[allow_redirects][url] = result  # Cache result
                return url, result
        tasks = [validate(u) for u in urls]
        results = await asyncio.gather(*tasks)
        return results

async def validate_urls_main2(urls, max_concurrency=MAX_CONCURRENCY):
    semaphore = asyncio.Semaphore(max_concurrency)
    async with aiohttp.ClientSession(headers=HEADERS) as session:
        async def validate(url):
            if url in RD_URL_CACHE_TP:
                return url, RD_URL_CACHE_TP[url]  # Return cached result
            async with semaphore:
                result = await async_validate_url_with_textpunctuation(session, url)
                if result[1] != 429:
                    RD_URL_CACHE_TP[url] = result  # Cache result
                return url, result
        tasks = [validate(u) for u in urls]
        results = await asyncio.gather(*tasks)
        return results

# --- Processing ---

async def get_links_df(posts_df, column='body'):

    # print(len(posts_df))
    posts_df['extracted_links'] = posts_df[column].dropna().apply(extract_links_from_text)
    # print(posts_df['extracted_links'].iloc[0])
    # print(posts_df['extracted_links'].iloc[-1])
    links_df = posts_df.explode('extracted_links').rename(columns={'extracted_links':'extracted_url'}).reset_index(drop=True)
    links_df = links_df[~links_df['extracted_url'].isna()].drop_duplicates().copy()
    # print(len(links_df))
    # print('egl', links_df)

    return await process_links(links_df)

async def process_links(links_df, retry_count=0, max_retries=5, backoff_factor=3):
    chop_index = list(links_df.columns).index('extracted_url') + 1
    # 1) Initial validation
    urls = links_df['extracted_url'].unique()
    # print(urls)
    results = await validate_urls_main(urls)
    # print(len(results))
    df_results = pd.DataFrame([[x[0], *x[1]] for x in results],
                              columns=['extracted_url', 'valid_1', 'status_1', 'processed_url_1']).drop_duplicates('extracted_url')
    # print(len(df_results))
    # links_df.to_hdf('data/links_df_test.h5', key='df', mode='w')
    # df_results.to_hdf('data/df_results_test.h5', key='df', mode='w')
    links_df = links_df.merge(df_results, on='extracted_url', how='left').drop_duplicates()
    # links_df.to_hdf('data/links_df_test1.h5', key='df', mode='w')
    # print(len(links_df))
    # 2) Reextract for invalid URLs
    invalid_urls = links_df.loc[links_df['valid_1']==False, 'extracted_url'].unique()

    if len(invalid_urls) > 0:
        # Create a mapping DataFrame of extracted_url -> reextracted_url
        invalid_urls_series = pd.Series(invalid_urls, name='extracted_url')
        reextracted_list = []
        for u in invalid_urls_series:
            reex_urls = reextract_links(u)
            for rurl in reex_urls:
                reextracted_list.append((u, rurl))

        if reextracted_list:
            reextracted_df = pd.DataFrame(reextracted_list, columns=['extracted_url', 'reextracted_url']).drop_duplicates('extracted_url')
            # reextracted_df.to_hdf('data/reextracted_df_test.h5', key='df', mode='w')
            # print('re', reextracted_df)
            # Validate reextracted URLs
            re_urls = reextracted_df['reextracted_url'].unique()
            # print('ru', re_urls)
            results2 = await validate_urls_main(re_urls)
            df_results2 = pd.DataFrame([[x[0], *x[1]] for x in results2],
                                       columns=['reextracted_url', 'valid_2', 'status_2', 'processed_url_2']).drop_duplicates('reextracted_url')
            # print('r2', df_results2)
            # Merge second attempt results into reextracted_df
            # df_results2.to_hdf('data/df_results2_test.h5', key='df', mode='w')
            reextracted_df = reextracted_df.merge(df_results2, on='reextracted_url', how='left').drop_duplicates()
            # print('re2', reextracted_df)
            # Now merge reextracted results back to links_df using 'extracted_url'
            # prin t('l0', links_df)
            # reextracted_df.to_hdf('data/reextracted_df_test2.h5', key='df', mode='w')
            links_df = links_df.merge(reextracted_df, on='extracted_url', how='left', suffixes=('', '_second')).drop_duplicates()
            # print('lr', links_df)
            # Check still invalid after reextraction
            # Consider invalid if original attempt and reattempt still fail
            # links_df.to_hdf('data/links_df_test2.h5', key='df', mode='w')
            # print('ldf', len(links_df))
            # print('re', len(re_urls))
            # print('r2', len(results2))
            # print('rdf', len(reextracted_df))
            still_invalid_df = links_df.loc[
                (links_df['valid_1']==False) &
                ((links_df['valid_2'].isna()) | (links_df['valid_2']==False))
            ][['reextracted_url']].drop_duplicates().copy()
            # print('si', still_invalid_df)
            if len(still_invalid_df) > 0:
                # Validate still invalid URLs with textpunctuation
                # We need to look up the reextracted URLs again for these still invalid URLs
                # or possibly re-run the logic if reextraction is required again.
                # For now, let's assume we can directly use 'reextracted_df' to get them.

                # Filter reextracted_df for those matching still_invalid original URLs
                re_urls_third = still_invalid_df['reextracted_url'].unique()
                # print('r3', re_urls_third)
                results3 = await validate_urls_main2(re_urls_third)
                df_results3 = pd.DataFrame([[x[0], *x[1]] for x in results3],
                                           columns=['reextracted_url', 'valid_3', 'status_3', 'processed_url_3']).drop_duplicates('reextracted_url')

                # Merge third attempt results
                still_invalid_df = still_invalid_df.merge(df_results3, on='reextracted_url', how='left').drop_duplicates()
                
                # Merge back into links_df by original_url
                links_df = links_df.merge(still_invalid_df, on='reextracted_url', how='left', suffixes=('', '_third')).drop_duplicates()

    # Now links_df contains:
    # - original extracted_url from the posts
    # - initial validation results (valid, status, processed_url)
    # - second attempt validation results (valid_re, status_re, processed_url_re) via reextraction
    # - third attempt validation results (valid_third, status_third, processed_url_third)

    # You can now decide on final validity and merge results back into posts if needed.

    # add cols to links_df, if not present
    cols = ['extracted_url', 'valid_1', 'status_1',
       'processed_url_1', 'reextracted_url', 'valid_2', 'status_2', 'processed_url_2',
       'valid_3', 'status_3', 'processed_url_3']
    for col in cols:
        if col not in links_df.columns:
            links_df[col] = None

    # get final validity, prefer the last valid url
    if len(links_df) > 0:
        links_df['end_processed_valid'] = links_df['valid_3'].fillna(
            links_df['valid_2'].fillna(
                links_df['valid_1']
            )
        )

        links_df['end_processed_status'] = links_df['status_3'].fillna(
            links_df['status_2'].fillna(
                links_df['status_1']
            )
        )
        
        links_df['end_processed_url'] = links_df['processed_url_3'].fillna(
            links_df['processed_url_2'].fillna(
                links_df['processed_url_1']
            )
        )
        
        # print('l', links_df)
        # get redirects for 3xx errors
        error3xxs = links_df[(links_df['status_1']>=300)&(links_df['status_1']<400)|
                             (links_df['status_2']>=300)&(links_df['status_2']<400)|
                             (links_df['status_3']>=300)&(links_df['status_3']<400)
                             ][['end_processed_url']].drop_duplicates().copy()

        rd_results = await validate_urls_main(error3xxs['end_processed_url'], retries=100000000, allow_redirects=True)
        rd_df = pd.DataFrame([[x[0], *x[1]] for x in rd_results],
                             columns=['end_processed_url', 'valid_rd', 'status_rd', 'redirected_url']
                             ).drop_duplicates('end_processed_url')
        # print('rd', rd_df)
                             
        links_df = links_df.merge(rd_df, on='end_processed_url', how='left').drop_duplicates()

        #  rerun for 429 errors
        error429s = links_df[(links_df['status_1']==429)|
                                (links_df['status_2']==429)|
                                (links_df['status_3']==429)|
                                (links_df['status_rd']==429)
                                ][links_df.columns[:chop_index]].drop_duplicates().copy()
        links_df = links_df[(links_df['status_1']!=429)&
                            (links_df['status_2']!=429)&
                            (links_df['status_3']!=429)&
                            (links_df['status_rd']!=429)
                            ].drop_duplicates()

        if len(error429s) > 0:
            if retry_count < max_retries:
                delay = backoff_factor ** retry_count
                print(f"429 errors: {len(error429s)}. Retrying after {delay} seconds...")
                await asyncio.sleep(delay)
                # Retry recursively with increased retry_count
                new_links = await process_links(error429s.copy(), retry_count=retry_count + 1, max_retries=max_retries, backoff_factor=backoff_factor)
                links_df = pd.concat([links_df, new_links]).drop_duplicates()
            else:
                print(f"Max retries reached for {len(error429s)} URLs. Skipping...")
                error429s_extra = links_df[(links_df['status_1']==429)|
                                (links_df['status_2']==429)|
                                (links_df['status_3']==429)|
                                (links_df['status_rd']==429)
                                ][links_df.columns[:chop_index]].drop_duplicates().copy()
                error429s = pd.concat([error429s, error429s_extra])
                links_df = links_df[(links_df['status_1']!=429)&
                            (links_df['status_2']!=429)&
                            (links_df['status_3']!=429)&
                            (links_df['status_rd']!=429)
                            ].drop_duplicates()
                # print(f"429 errors: {len(l1429)} -> {len(error429s)}")
                # save 429s
                if len(error429s) > 0:
                    print("Saving 429s")
                    error429s.to_hdf('data/429s.h5', key='df', mode='a')


        links_df['final_valid'] = links_df['valid_rd'].fillna(links_df['end_processed_valid'])
        links_df['final_status'] = links_df['status_rd'].fillna(links_df['end_processed_status'])
        links_df['final_url'] = links_df['redirected_url'].fillna(links_df['end_processed_url'])

        print("Saving cache. Total size: ", len(RD_URL_CACHE_P[True]) + len(RD_URL_CACHE_P[False])
                + len(RD_URL_CACHE_TP))
        with open('.temp/rd_url_cache_p.pkl', 'wb') as f:
            pickle.dump(RD_URL_CACHE_P, f)
        with open('.temp/rd_url_cache_tp.pkl', 'wb') as f:
            pickle.dump(RD_URL_CACHE_TP, f)

        return links_df.drop_duplicates().reset_index(drop=True)
    else:
        return links_df.drop_duplicates().reset_index(drop=True)


# --- Main Asynchronous Processing Function ---
async def process_comments():
    """Main asynchronous function to process comments."""
    global MAX_CONCURRENCY

    # Combine all comments data
    comments = pd.concat([pd.read_hdf(f'data/comments_{x}.h5') for x in range(1, 5)]).reset_index(drop=True)

    outcols = ['id', 'extracted_url', 'valid_1', 'status_1',
               'processed_url_1', 'reextracted_url', 'valid_2', 'status_2',
               'processed_url_2', 'valid_3', 'status_3', 'processed_url_3',
               'end_processed_valid', 'end_processed_url',
               'valid_rd', 'status_rd', 'redirected_url', 'final_valid',
               'final_status', 'final_url']
    
    size = len(comments) // 100
    commentlinks = []

    for batch in range(0, len(comments), size):
        print(f"Batch {batch // size + 1}", size, datetime.datetime.now())
        try:
            commentlinks.append(pd.read_hdf(f'data/commentlinks_batches.h5', key=f'/batch_{batch // size + 1}'))
        except (FileNotFoundError, KeyError) as ex:
            print(ex)
            while True:
                try:
                    links_df = await get_links_df(comments.iloc[batch:batch + size].copy(), column='body')
                    MAX_CONCURRENCY = min(200, int(round(MAX_CONCURRENCY * (2 ** 0.6), 0)))
                    break
                except MemoryError as ex:
                    print(ex)
                    MAX_CONCURRENCY = MAX_CONCURRENCY // 2
            if len(links_df) > 0:
                print('Saving batch...')
                links_df[outcols].to_hdf(f'data/commentlinks_batches.h5', key=f'/batch_{batch // size + 1}', mode='a')
                commentlinks.append(links_df[outcols])

    # Combine all batches into a single DataFrame
    commentlinks = pd.concat(commentlinks).reset_index(drop=True)

    if os.path.exists('data/commentlinks.h5'):
        os.remove('data/commentlinks.h5')

    commentlinks.to_hdf('data/commentlinks.h5', key='df', mode='w')

# --- Entry Point ---
if __name__ == "__main__":

    # Run the main async function
    asyncio.run(process_comments())
