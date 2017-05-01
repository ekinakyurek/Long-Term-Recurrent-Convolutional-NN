#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import sys
import math
import glob

if len(sys.argv) != 3:
    exit(0)

#==========================================================#
outFile= open('bleu_out.txt', 'w')
#==========================================================#

#==========================================================#
candidate = sys.argv[1]
candidateHand = open(candidate, 'r')
#==========================================================#

#==========================================================#
reference = sys.argv[2]
if os.path.isdir(reference):
    if sys.argv[2].endswith('/'):
        docs = glob.glob(reference + '*.txt')
    else:
        docs = glob.glob(reference + '/*.txt')
else:
    docs = list()
    docs.append(reference)

reference = dict()
for doc in docs:
    reference[doc] = open(doc, 'r')
#==========================================================#
def getBigram(words):
    temp = list()
    for i in range(len(words) - 1):
        temp.append(words[i] + u' ' + words[i+1])
    return temp
#=======================================================================#
def getTrigram(words):
    temp = list()
    for i in range(len(words) - 2):
        temp.append(words[i] + u' ' + words[i + 1] + u' ' + words[i + 2])
    return temp
#=======================================================================#
def getQuadgram(words):
    temp = list()
    for i in range(len(words) - 3):
        temp.append(words[i] + u' ' + words[i + 1] + u' ' + words[i + 2] + u' ' + words[i + 3])
    return temp
#=======================================================================#
def getFreq(freq, unique, line, ref):
    for word in unique:
        match = min(line.count(word), ref.count(word))
        if not word in freq:
            freq[word] = match
        elif freq[word] < match:
            freq[word] = match
    return freq
#=============================================================#
def computeGram(line, counter):
    freq, freq2, freq3, freq4 = dict(), dict(), dict(), dict()
    # For UNIGRAM
    line = line.lower()
    line = line.split()
    unique = list(set(line))
    denominator = len(line)
    C_EXP.append(denominator)
    # For BIGRAM
    line2= getBigram(line)
    denominator2 = len(line2)
    unique2 = list(set(line2))
    # For TRIGRAM
    line3= getTrigram(line)
    denominator3 = len(line3)
    unique3 = list(set(line3))
    # For QUADGRAM
    line4 = getQuadgram(line)
    denominator4 = len(line4)
    unique4 = list(set(line4))
    #========================================================#
    i = 1
    for doc in docs:
        ref = reference[doc].readline()
        ref = ref.strip().decode('utf-8', 'ignore')
        ref = ref.lower()
        ref = ref.split()
        if not i in R_EXP:
            R_EXP[i] = {counter : len(ref)}
        else:
            R_EXP[i][counter] = len(ref)
        i  += 1
        #====================================================#
        # UNIGRAM
        getFreq(freq, unique, line, ref)
        n1 = sum(freq.values())
        d1 = denominator
        #====================================================#
        # BI-GRAM
        ref2 = getBigram(ref)
        getFreq(freq2, unique2, line2, ref2)
        n2 = sum(freq2.values())
        d2 = denominator2
        #====================================================#
        # TRI-GRAM
        ref3 = getTrigram(ref)
        getFreq(freq3, unique3, line3, ref3)
        n3 = sum(freq3.values())
        d3 = denominator3
        #====================================================#
        # QUAD-GRAM
        ref4 = getQuadgram(ref)
        getFreq(freq4, unique4, line4, ref4)
        n4 = sum(freq4.values())
        d4 = denominator4
    #========================================================#
    return n1, d1, n2, d2, n3, d3, n4, d4
#============================================================#

#=============================================================================#
R_EXP = dict()
C_EXP = list()
weight = 0.25
a, b, c, d = 0, 0, 0, 0
p, q, r, s = 0, 0, 0, 0
# Processing Step
#========================================================#
counter = 0
for line in candidateHand:
    counter += 1
    line = line.strip().decode('utf-8', 'ignore')
    n1, d1, n2, d2, n3, d3, n4, d4 = computeGram(line, counter)
    a, b, c, d = a + n1, b + n2, c + n3, d + n4
    p, q, r, s = p + d1, q + d2, r + d3, s + d4
#========================================================#
p1 = float(a)/p
p2 = float(b)/q
p3 = float(c)/r
p4 = float(d)/s
p = 0
try:
    p = math.log(p1)
except:
    pass
try:
    p += math.log(p2)
except:
    pass
try:
    p += math.log(p3)
except:
    pass
try:
    p += math.log(p4)
except:
    pass
p = weight * p

#========================================================#
R = 0
for index in range(len(C_EXP)):
    temp = None
    DIFF = None
    for key in R_EXP:
        if not DIFF or abs(R_EXP[key][index + 1] - C_EXP[index]) < DIFF:
            DIFF = abs(R_EXP[key][index + 1] - C_EXP[index])
            temp = R_EXP[key][index + 1]
    # Added closest value for each sentence
    R += temp

#========================================================#
C_EXP = sum(C_EXP)
if C_EXP > R:
    BP = 1.0
else:
    BP = math.exp(1 - float(R)/C_EXP)
BLEU = BP * math.exp(p)
outFile.write(str(BLEU))
outFile.write('\n')
#==========================================================#
outFile.close()
candidateHand.close()
for fhand in reference.values():
    fhand.close()
#==========================================================#
