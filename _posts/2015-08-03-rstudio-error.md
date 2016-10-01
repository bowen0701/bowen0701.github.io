---
layout: post
title: "Loading R Packages Error on RStudio"
date: 2015-08-03
---

My R version is 3.2.1, on Macbook Air, but I cannot load some useful packages (including `forecast`, `TSA` and `mgcv` packages) on RStudio; attached is the error message:

> Error : .onAttach failed in attachNamespace() for 'forecast', details: call: formatDL(nm, txt, indent = max(nchar(nm, "w")) + 3) error: incorrect values of 'indent' and 'width'

Then I spent so much time on this issue this afternoon (2015/08/03) and finally found a [StackOverFlow article](http://stackoverflow.com/questions/19086111/package-mgcv-could-not-be-loaded-only-in-rstudio) which provided two methods to solve this serious problem (Yes, it's serious, I mean it.) Now I summarize both methods as the following.

## Method 1

We can solve this problem "temporariy" by the following shell script for removing the RStudio user setting,

```shell
rm -r ~/.rstudio-desktop/monitored/user-settings/
```

- then load `forecast` package (and `TSA`, `mgcv`, among others) first,
- finally reset my RStudio setting.

Why I say it's temporarily is because every time when I restart RStudio, I must repeat the above procedure. Otherwise those packages would not be able to upload.

## Method 2
Amazingly, the following seemingly ridiculous procedure can completely solve the problem:
- We can "literally" resize (i.e. increase the size of) the window of Console or the one consisting of Environment or History on RStudio,
- then we would be able to load the package at any time.

It seems Method 2 is easier than Method 1. Honestly, I don't know why Method 2 is applicable, but it does solve this unsuccessfully loading package problem and make me able to go back to work. :-)
