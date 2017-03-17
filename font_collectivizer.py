#! /usr/bin/env python
# -*- coding: utf-8
import wget
import time
import requests
import os
import shutil
import subprocess

from bs4 import BeautifulSoup
from pprint import pprint
from zipfile import ZipFile


def write_to_log(msg, logname):
    with open(logname,'a') as log:
        log.write(msg+'\n')


def get_html(url):
    fname = wget.download(url)
    with open(fname,'r') as f:
        html = f.read()
    time.sleep(0.1)
    return html


def make_soup(html_doc):
    return BeautifulSoup(html_doc, 'html.parser')


def find_font_colls(soup):
    font_colls = []
    for link in soup.find_all('a'):
        url = link.get('href')
        if url.split('-')[-1] == 'fonts.html' and url[:4] == 'http':
            font_colls.append(url)
    return font_colls


def find_local_fonts(soup):
    fonts = []
    for link in soup.find_all('a'):
        url = link.get('href')
        if url.split('.')[-1] == 'zip':
            fonts.append(url)
    return fonts


def get_initial_font_collections():

    base_url = "" # nope
    # Get html_doc from URL and make soup
    html_doc = get_html(base_url)
    soup = make_soup(html_doc)

    # Find the font collection URLs and local fonts.
    font_colls = find_font_colls(soup)

    return font_colls


def get_all_font_coll_urls(font_coll_url, fonts, logname):
    k = 1
    kmax = 1000
    all_urls = []
    while True:
        addme = "&page={0}&items=10".format(k)
        local_url = font_coll_url+addme
        # Keep us updated on the progress being made.
        print(local_url)
        status = requests.head(font_coll_url+addme).status_code
        if status == 404:
            break
        # Go ahead and scrape the font zip URLs since we know that page exists.
        n_tries = 2
        for _ in range(n_tries):
            try:
                get_local_zip_urls(local_url, fonts, logname)
                break
            except:
                time.sleep(1.0)

        all_urls.append(local_url)
        k += 1
        # time.sleep(0.1)
        if k > kmax:
            break

    return all_urls


def get_local_zip_urls(url, fonts, logname):
    html_doc = get_html(url)
    soup = make_soup(html_doc)
    font_urls = find_local_fonts(soup)
    for x in font_urls:
        if x not in fonts:
            print(x)
            fonts[x] = 1
            write_to_log(x, logname)


def extract(extract_dir):
    urls = extract_dir + "-font-urls.txt"

    font_dir = extract_dir
    os.chdir(font_dir)

    fh = open(urls, 'r')
    urls = fh.read().split('\n')
    fh.close()


    pprint(urls[0])
    pprint(urls[-1])
    pprint(len(urls))
    urls = urls[1:-1]

    # This needs to be an absolute path.
    collection = font_dir + '-collection'

    os.mkdir(collection)
    abscoll = os.path.abspath(collection)
    print(abscoll)

    for x in urls:
        # Open the zipfile.
        fname = x.split('/')[-1]
        tmp = './tmp'
        os.mkdir(tmp)
        print(x, fname)
        try:
            # Try to extract it into the temp directory.
            zfile = ZipFile(fname,'r')
            zfile.extractall(tmp)
        except:
            # if it's broken, don't stress, just move on to the next one.
            shutil.rmtree(tmp)
            continue

        os.chdir(tmp)
        for xx in os.listdir('.'):
            if os.path.isdir(xx):
                for xxx in os.listdir(xx):
                    ext = os.path.splitext(xxx)[-1]
                    src = os.path.abspath(xx)+'/'+xxx
                    if  ext == '.otf' or ext == '.ttf':
                        print(src)
                        shutil.copy(src.encode('ascii','ignore'), abscoll)
            else:
                ext = os.path.splitext(xx)[-1]
                src = os.path.abspath(xx)
                if ext == '.otf' or ext == '.ttf':
                    print(src)
                    shutil.copy(src.encode('ascii','ignore'), abscoll)
        os.chdir('..')
        shutil.rmtree('./tmp')


    # Tired of doing this by hand.
    os.chdir('..')
    d = extract_dir
    dst = d + '.zip'
    src = d + '/' + d + '-collection'
    cmd = 'zip -r {0} {1}'.format(dst, src)
    p = subprocess.Popen(cmd, shell=True)
    p.communicate()





def main():
    font_colls = get_initial_font_collections()
    annotated_font_colls = [(x, k) for k, x in enumerate(font_colls)]
    pprint(annotated_font_colls)
    print(len(font_colls))
    print("Starting...")
    # Go get 'em.
    for style in font_colls:
        font_zip_urls = {}
        print("Okay we're doing the collection found at: ")
        stylename = style.split('/')[-1].split('.')[0]
        print(stylename)
        if not os.path.exists(stylename):
            os.mkdir(stylename)
        os.chdir(stylename)
        logname = stylename + '-font-urls.txt'
        write_to_log("# Collection for {}".format(stylename), logname)

        n_tries = 2
        for _ in range(n_tries):
            try:
                all_urls = get_all_font_coll_urls(style, font_zip_urls, logname)
                break
            except:
                pass


        urls = font_zip_urls.keys()
        pprint(urls)
        N = len(urls)
        for k, x in enumerate(urls):
            f = x.split('/')[-1]
            if not os.path.exists(f):
                for _ in range(n_tries):
                    try:
                        fname = wget.download(x)
                        break
                    except:
                        pass
            print(f)
            print("{} / {}".format(k, N))
        # Go back up one level or you're going to have a chain of directories
        # inside directories.
        os.chdir('..')

        # Unzip, collect the .otf and .ttf files and zip again.
        extract(stylename)



if __name__ == "__main__":
    main()
