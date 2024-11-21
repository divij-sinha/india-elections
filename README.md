# India Elections Project

## Form 20 scraper

`form20s/form20_dler.py`

Simple script to download all Form 20s from all the Election Commission of India state websites.

Form 20 refers to the results of the Parliamentary Constituency Elections. It is a PDF file that contains the details of the election results. Form 20 Part 1 is the Parliamentary Constituency level results and Form 20 Part 2 is the Assembly Constituency level results.

Fatima TODO: Add as many states as possible! I have added a few states, and two different methods to do so.   

1. String formatting based - when the url for each pdf is very similar and you can deterministically figure out the next link. 
2. Page based - when the urls for each pdf are on the same page, just give the page and download away. You can filter links in or out using strings. (`positive`: keep link if present, `negative`: remove link if present).

Ideally, you will be able to partial more states in the same way. I would say, its better to find more states that work the same way and skip others, than to spend a long time on a single state that works a little differently.

Here is a small tracker to keep track of which states have been added and which are pending. I have marked with an `*` the states that I think might be easier to do. This is based on nothing but my intuition, so feel free to ignore it.

Suggested work flow:

1. Search for "<state> Form 20 2024" or "<state> election commission". These websites will usually have the string "ceo" (Chief Election Officer) or "elections" in the url, and will always end with "gov.in"
1. Look for the Form 20s on the website. They are usually under "Election Results" or "Election Statistics" or "Election Reports". They might say "Form 20" or "Parliamentary Constituency Results" or "Lok Sabha Results". We want the ones from 2024
1. Figure out the best way to scrape them. If you think it fits well with our methods, go ahead and do it. Otherwise, skip.
1. There are a few states that can be done manually, as they are not really states and will have 1-2 files at most. Feel free to download them manually and add them to the tracker, no need to scrape when we dont need to.

Tracker:
https://docs.google.com/spreadsheets/d/1Cw4NaWeBzJeGCIuUjVOzHlK2HdCz9PFNvX_SAes9Pyk/edit?usp=sharing