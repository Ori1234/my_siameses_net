# -*- coding: utf-8-*-
import os
import Image
import ImageDraw
import ImageFont
from random import randint
import ImageFilter
letters=[u'א',u'ב',u'ג',u'ד',u'ה',u'ו',u'ז',u'ח',u'ט',u'י',u'כ',u'ך',u'ל',u'מ',u'ם',u'נ',u'ן',u'ס',u'ע',u'פ',u'ף',u'צ',u'ץ',u'ק',u'ר',u'ש',u'ת']
letters=[u'אורי']

#letters=list('abcdefghijklmnopABCDEFGHIJKLMNOPQRSTUVWXYZ')
letters_folder="words_RGB"
#letters=[u'א']
#path='/usr/share/fonts/hebrew'
fonts=[]
#for dirpath,dirnames,filenames in os.walk(path):
#	for filename in [f for f in filenames if f.endswith(".ttf")]:
#		fontPath=os.path.join(dirpath,filename)
#		fonts.append(fontPath)

list_fonts='hebrew_fonts.txt'
file = open(list_fonts, 'r')
for fontPath in file:
	if fontPath.lower().endswith(".ttf"):
		print '#this ', fontPath
	else:
#		print(fontPath)
		fonts.append(fontPath.rstrip('\n'))


def reverse(str):
	return str[::-1]

import sys
#sys.exit()
if not os.path.exists(letters_folder):
	os.makedirs(letters_folder)

for fontPath1 in fonts:
	
	#print(fontPath1)	
	fontPath=fontPath1.strip().strip(':')
#	print(fontPath)	
#	fontName=fontPath.split('/')[-1][:-5]
	fontName=os.path.splitext(os.path.basename(fontPath))[0]
	print(fontName)
	for l in letters:
	#l='א'
	#print(l)
		r=reverse(l) #TODO how to align the ImageFont to right?
	  	size=(64,64)
		fontsize = 1  # starting font size	
		# portion of image width you want text width to be
		img_fraction = 1
		margin=5
		font = ImageFont.truetype(fontPath, fontsize)
		while font.getsize(r)[0] < size[0]-2*margin and font.getsize(r)[1] < size[1]-2*margin:
		    # iterate until the text size is just larger than the criteria
		    fontsize += 1
		    font = ImageFont.truetype(fontPath, fontsize)

		# optionally de-increment to be sure it is less than criteria
		fontsize -= 1
		font = ImageFont.truetype(fontPath,fontsize)

		#print 'final font size',fontsize
				
		lett_size=font.getsize(r)
		borders=(size[0]-lett_size[0],size[1]-lett_size[1])	
		offset=font.getoffset(r)
		offset=((-offset[0]+borders[0])/2,(-offset[1]+borders[1])/2)
		#print(size)
#		img=Image.new("RGBA",size,"white")
#		img=Image.new("L",size,"white")
		img=Image.new("RGB",size,"white")
		draw=ImageDraw.Draw(img)
		draw.text(offset,r,"black",font=font)
		
#		for i in range(1,300): #Draw random points
#			draw.point((randint(0,size[0]),randint(0,size[1])),fill=0)
		#blur
#		img=img.filter(ImageFilter.GaussianBlur(radius=1))
		
		img.save(letters_folder+"/"+l+"_"+fontName+".png")
#		img.save(letters_folder+"/"+l+"_"+fontName+str(offset)+str(size)+".png")

