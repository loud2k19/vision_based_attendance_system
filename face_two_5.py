import cv2
import sqlite3
import time, threading
import datetime
import urllib.request as urllib2
import threading
import numpy as np

import multiprocessing
cam = cv2.VideoCapture(2)
cam2 = cv2.VideoCapture(0)


inn=[1,1,1,1,1,1,1,1,1,1]
out=[0,0,0,0,0,0,0,0,0,0]
att=[0,0,0,0,0,0,0,0,0,0]
idx=[1,2,3,4,5,6,7,8,9,10]

global names
global my
global count
global z1
global z2
z1=[1,1,1,1,1,1,1,1,1,1,1]
z2=[1,1,1,1,1,1,1,1,1,1,1]

count = 0
names = ['None', 'himesh','abishek','lekya','harini']

def final():
    global count
    global my	
    i=1
    while(i<2):
     att[i]=int(not(inn[i]^out[i]))
     att[i]=str(att[i])
     #dynamic_data(i,names[i],att[i])
     i=i+1
    count=count+1
    
    peri()
    threading.Timer(3.0,final).start()

def peri():
    global count	
    if count==2:
       i=1
       z='math'
       while(i<5):
         att[i]=str(att[i])
         dynamic_data(i,names[i],att[i],z)
         i=i+1


    elif( count == 4):
        i=1			 
        z='mpmc'
        while(i<5):
             att[i]=str(att[i])
             dynamic_data(i,names[i],att[i],z)
             i=i+1

    elif (count ==6):
        i=1
        z='dsp'
        while(i<5):
          att[i]=str(att[i])
          dynamic_data(i,names[i],att[i],z)
          i=i+1
        
    elif( count ==8):
        i=1
        z='tlw'
        while(i<5):
          att[i]=str(att[i])
          dynamic_data(i,names[i],att[i],z)
          i=i+1
        


def dynamic_data(id,name,value,period):
    #urllib3.urlopen("http://www.educ8s.tv/weather/add_data.php?temp=" + temp + "&hum=" + hum + "&pr=" + press ).read()

    if period=='math':
    	g="https://pi22.000webhostapp.com/add_data.php?id=" + str(id)
    	h=g+"&name=" + name
    	l=h + "&attend="
    	y=l + value
    	print(y)

    	urllib2.urlopen(y ).read()
    if period=='mpmc':
    	g1="https://pi22.000webhostapp.com/add_mpmc.php?id=" + str(id)
    	h1=g1+"&name=" + name
    	l1=h1 + "&attend="
    	y1=l1 + value
    	print(y1)

    	urllib2.urlopen(y1 )#.read()
    if period=='dsp':
    	g="https://pi22.000webhostapp.com/add_dsp.php?id=" + str(id)
    	h=g+"&name=" + name
    	l=h + "&attend="
    	y=l + value
    	print(y)

    	urllib2.urlopen(y ).read()
    if period=='tlw':
    	g="https://pi22.000webhostapp.com/add_tlw.php?id=" + str(id)
    	h=g+"&name=" + name
    	l=h + "&attend="
    	y=l + value
    	print(y)

    	urllib2.urlopen(y ).read()

def f(id1):
	
  	n=1000000000
  	while n>0:
                 n=n-1
                 if n==0:
                       global z1
                       z1[id1]=1
                       print('out')
                       break

def f1(id1):
	
        n=10000000
        while n>0:
                n=n-1
                if n==0:
                  global z2
                  z2[id1]=1
                  print(n)
                  break
#def pause():
	

status=[0,0,0,0]
"""conn=sqlite3.connect('Attendance.db')
c=conn.cursor()"""
now = datetime.datetime.now()
today8am=now.replace(hour=21,minute=0,second=0)

recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read( 'trainer/TRY_2.yml' )
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier( cascadePath )
faceCascade2 = cv2.CascadeClassifier( cascadePath )

font = cv2.FONT_HERSHEY_SIMPLEX

cam.set( 3, 640 )
cam.set( 4, 480 )

cam2.set( 3, 640 )
cam2.set( 4, 480 )

minW = 0.1 * cam2.get( 3 )
minH = 0.1 * cam2.get( 4 )

minW = 0.1 * cam.get( 3 )
minH = 0.1 * cam.get( 4 )




final()
while True:
    # Capture frame-by-frame
    ret0, img = cam.read()
    ret1, img2 = cam2.read()

    img = cv2.flip( img, 1 )
    img2 = cv2.flip( img2, 1 )

    gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
    gray2 = cv2.cvtColor( img2, cv2.COLOR_BGR2GRAY )


    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces :
        cv2.rectangle( img, (x, y), (x + w, y + h), (0, 255, 0), 2 )
        id1, confidence = recognizer.predict( gray[y :y + h, x :x + w] )
       # if now<today8am :
        if (confidence < 100) :
                 if id1==3:
                  if z1[id1]==1 :    
                   inn[id1]=int(not(inn[id1]))
                   id2 = names[id1]+" {0}%".format( round( 100 - confidence ) )
		   #print(names[id1]+"out")
		   #i1=threading.Timer(3,pause())
		   #i1.start()
                   print("2")
                   cv2.putText( img, str( id2 ), (x + 5, y - 5), font, 1, (255, 255, 255), 2 )
                   z1[id1]=0
                   p=multiprocessing.Process(target=f,args=(id1,))
                   p.start()  
        if (confidence < 100) :
                 if id1 ==1:
                  if z1[id1]==1 :    
                   inn[id1]=int(not(inn[id1]))
                   id2 = names[id1]+" {0}%".format( round( 100 - confidence ) )
		   #print(names[id1]+"out")
		   #i1=threading.Timer(3,pause())
		   #i1.start()
                   print("2")
                   cv2.putText( img, str( id2 ), (x + 5, y - 5), font, 1, (255, 255, 255), 2 )
                   z1[id1]=0
                   p=multiprocessing.Process(target=f,args=(id1,))
                   p.start()  
        if (confidence < 100) :
                 if id1==2:
                  if z1[id1]==1 :    
                   inn[id1]=int(not(inn[id1]))
                   id2 = names[id1]+" {0}%".format( round( 100 - confidence ) )
		   #print(names[id1]+"out")
		   #i1=threading.Timer(3,pause())
		   #i1.start()
                   print("2")
                   cv2.putText( img, str( id2 ), (x + 5, y - 5), font, 1, (255, 255, 255), 2 )
                   z1[id1]=0
                   p=multiprocessing.Process(target=f,args=(id1,))
                   p.start()  
        if (confidence < 100) :
                 if id1==4:
                  if z1[id1]==1 :    
                   inn[id1]=int(not(inn[id1]))
                   id2 = names[id1]+" {0}%".format( round( 100 - confidence ) )
		   #print(names[id1]+"out")
		   #i1=threading.Timer(3,pause())
		   #i1.start()
                   print("2")
                   cv2.putText( img, str( id2 ), (x + 5, y - 5), font, 1, (255, 255, 255), 2 )
                   z1[id1]=0
                   p=multiprocessing.Process(target=f,args=(id1,))
                   p.start()  		   	
                   
        else :
                id = "unknown"
                confidence = "  {0}%".format( round( 100 - confidence ) )
                cv2.putText( img, str( id ), (x + 5, y - 5), font, 1, (255, 255, 255), 2 )
             #cv2.putText( img, str( id ), (x + 5, y - 5), font, 1, (255, 255, 255), 2 )
    cv2.imshow( 'in', img)

    faces2 = faceCascade2.detectMultiScale(
        gray2,
        scaleFactor=1.08,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces2 :
        cv2.rectangle( img2, (x, y), (x + w, y + h), (0, 255, 0), 2 )
        id1, confidence = recognizer.predict( gray2[y :y + h, x :x + w] )
        #if now<today8am :
        if (confidence < 100):
                 if id1==1:
                  if z2[id1]==1 : 
                   out[id1]=int(not(out[id1]))
                   id2 = names[id1]+" {0}%".format( round( 100 - confidence ) )
                   #print(names[id1] + "out")   
		   #z1=threading.Timer(3,pause)		  
		   #z1.start() 
                   print(id1)
                   #threading.Timer(30.0,final).start()
                   cv2.putText( img2, str( id2 ), (x + 5, y - 5), font, 1, (255, 255, 255), 2 )
                   z2[id1]=0
                   p1=multiprocessing.Process(target=f1,args=(id1,))
                   p1.start()
        if (confidence < 100):
                 if id1==2:
                  if z2[id1]==1 : 
                   out[id1]=int(not(out[id1]))
                   id2 = names[id1]+" {0}%".format( round( 100 - confidence ) )
                   #print(names[id1] + "out")   
		   #z1=threading.Timer(3,pause)		  
		   #z1.start() 
                   print(id1)
                   #threading.Timer(30.0,final).start()
                   cv2.putText( img2, str( id2 ), (x + 5, y - 5), font, 1, (255, 255, 255), 2 )
                   z2[id1]=0
                   p1=multiprocessing.Process(target=f1,args=(id1,))
                   p1.start()
        if (confidence < 100):
                 if id1==3:
                  if z2[id1]==1 : 
                   out[id1]=int(not(out[id1]))
                   id2 = names[id1]+" {0}%".format( round( 100 - confidence ) )
                   #print(names[id1] + "out")   
		   #z1=threading.Timer(3,pause)		  
		   #z1.start() 
                   print(id1)
                   #threading.Timer(30.0,final).start()
                   cv2.putText( img2, str( id2 ), (x + 5, y - 5), font, 1, (255, 255, 255), 2 )
                   z2[id1]=0
                   p1=multiprocessing.Process(target=f1,args=(id1,))
                   p1.start()
        if (confidence < 100):
                 if id1==4:
                  if z2[id1]==1 : 
                   out[id1]=int(not(out[id1]))
                   id2 = names[id1]+" {0}%".format( round( 100 - confidence ) )
                   #print(names[id1] + "out")   
		   #z1=threading.Timer(3,pause)		  
		   #z1.start() 
                   print(id1)
                   #threading.Timer(30.0,final).start()
                   cv2.putText( img2, str( id2 ), (x + 5, y - 5), font, 1, (255, 255, 255), 2 )
                   z2[id1]=0
                   p1=multiprocessing.Process(target=f1,args=(id1,))
                   p1.start()
                   

        else :
                id = "unknown"
                confidence = "  {0}%".format( round( 100 - confidence ) )
                cv2.putText( img2, str( id ), (x + 5, y - 5), font, 1, (255, 255, 255), 2 )

    cv2.imshow( 'out', img2)

    key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
     break
# When everything is done, release the capture
cam.release()
cam2.release()
cv2.destroyAllWindows()

