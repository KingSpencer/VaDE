#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 21:49:55 2019

@author: crystal
"""
import os

def createOutputFolderName(outputPath, Kmax, dataset, epoch, batch_iter, scale, batchsize, sf=0.1, rep=1):
    
    if not isinstance(Kmax, str):
        Kmax = str(Kmax)
    if not isinstance(dataset, str):
        raise ValueError("The input for dataset should be a string")
    if not isinstance(epoch, str):
        epoch = str(epoch)
    if not isinstance(batch_iter, str):
        batch_iter = str(batch_iter)
    if not isinstance(scale, str):
        scale = str(scale)
    if not isinstance(batchsize, str):
        batchsize = str(batchsize)
    if not isinstance(rep, str):
        rep = str(rep)
    if not isinstance(rep, str):
        sf = str(sf)
    
    folderName = 'Kmax' + Kmax + dataset + 'epoch' + epoch + 'batch_iter' + batch_iter + 'scale' + scale + 'bs' + batchsize + 'rep'+ rep + 'sf'+sf    
    wholeFolderName = os.path.join(outputPath, folderName)
    
    if os.path.exists(wholeFolderName):
        ## check if this folder exisits in outputPath,
        raise ValueError("The folder has already exisits")
    else:
        ## if not, create a folder with folderName in outputPath
        os.makedirs(wholeFolderName)
        
    return wholeFolderName
    
def createFullOutputFileName(wholeFolderName, fileName):
    if not isinstance(fileName, str):
        raise ValueError("The fileName should be of string type")
    else:
        fullFileName = os.path.join(wholeFolderName, fileName)
    return fullFileName
        
    
    
    
    
    
    
    
    
    
    