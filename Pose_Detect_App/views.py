from Pose_Detect_App.functions import handle_uploaded_file
from Pose_Detect_App.forms import VideoForm, AnalyseConfirm, Button1, Button2, Button3
from App_Script.pose import detect
from django.shortcuts import render
from django.http import HttpResponse
from django.http import StreamingHttpResponse
from time import sleep
import random
import json

file = ''

def index(request):
    global file

    if request.method == "POST":
        if (request.POST['ins'] == 'upload'):

            path = handle_uploaded_file(request.FILES['file'])  
            file = request.FILES['file']._name
            print(file)

            return HttpResponse('uploaded|' + path)  

        elif (request.POST['ins'] == 'analyse'):
            
            link = 'Pose_Detect_App/upload/' + file

            # return StreamingHttpResponse(detect(link))
            return StreamingHttpResponse(iterator())

        elif (request.POST['ins'] == 'button1'):
            return HttpResponse('Button1 Pressed')

        elif (request.POST['ins'] == 'button2'):
            return HttpResponse('Button2 Pressed')

        elif (request.POST['ins'] == 'button3'):
            return HttpResponse('Button3 Pressed')
    else:
        video = VideoForm()
        analyse = AnalyseConfirm()
        but1 = Button1()
        but2 = Button2()
        but3 = Button3()
        return render(request, "index.html", {'video': video, 'analyse': analyse, 'but1': but1, 'but2': but2, 'but3': but3})

def iterator():
    x = 100
    y = 0
    total = x
    fps = 30
    while (x > 0):
        y += 1
        x -= 1
        time = y/fps
        total_time = total/fps
        sleep(0.1)
        out = {
            "time": time,
            "total_time": total_time,
            "fps": fps
        }
        yield json.dumps(out) + '|'