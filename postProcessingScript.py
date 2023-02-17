import glob
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
import os, json
from tqdm import tqdm


parser = ArgumentParser()
parser.add_argument("-i", "--input", dest="input", help="Input txt file to begin the post processing with")
parser.add_argument("-o", "--output", dest="output", help="Output file txt file aftet the post processing")
parser.add_argument("-e", "--errorPath", dest="errorLogs", help="Errors Logs ")
args = parser.parse_args()

inFile = args.input
outFile = args.output
errLogs = args.errorLogs

def check(expression):

    open_tup = tuple('({[')
    close_tup = tuple(')}]')
    map = dict(zip(open_tup, close_tup))
    queue = []

    for i in expression:
        if i in open_tup:
            queue.append(map[i])
        elif i in close_tup:
            if not queue or i != queue.pop():
                return "Unbalanced"
    if not queue:
        return "Balanced"
    else:
        return "Unbalanced"
count =0
def postProcessing(line):
    global count
    count+=1
    # print(count)
    words = line.split(" ")
    if(words[0]!="@Test"):
        words.insert(0,"@Test")

    #append test if not
    id =  words.index("void")
    if(words[id+1][:4]!="test"):
        words[id+1] = "test"+words[id+1]

    status = check(line)
    if(status=="Unbalanced"):
        # print("Here")
        elem=None
        for word in words[::-1]:
            if(word[-1]==";"):
                elem = word
                break
        if(elem!=None):
            index_pos = len(words) - words[::-1].index(elem) - 1
            words = words[:index_pos+1]
            words.append("}")
        else:
            with open(errLogs,"a+") as f:
                f.write(" ".join(words));
            return None

    output = " ".join(words)
    return output

with open(inFile,'r') as inputF:
    content = inputF.readlines()
    for row in tqdm(content, desc="Processing Completion Status"):
        words = list(row.split(" "))
        if(len(words)==1):
            with open(errLogs,"a+") as f:
                f.write(" ".join(words));
            continue
        outPart = postProcessing(row)
        if(outPart==None):
            continue
        with open(outFile,'a+') as outF:
            if(outPart[-1]!="\n"):
                outF.write(outPart+"\n")
            else:
                outF.write(outPart)
