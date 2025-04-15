from django.shortcuts import render
import pymysql
from datetime import datetime
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
import cv2
import numpy as np
import pytesseract
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO

global username, disease, aadhar_data, aadhar_name, medical_data, medical_name
labels = ['Aadhar_no', 'DOB', 'Gender', 'Name']
#yolo confidence threshold to detect hand signs
CONFIDENCE_THRESHOLD = 0.50
GREEN = (0, 255, 0)
yolo_model = YOLO("model/best.pt")
print("Yolo11 Model Loaded")

#function to detect aadhar card
def getAadharNo(frame):
    global yolo_model
    detections = yolo_model(frame)[0]
    label = None
    aadhar_no = ""
    for data in detections.boxes.data.tolist():
        confidence = data[4]
        cls_id = data[5]
        if float(confidence) >= CONFIDENCE_THRESHOLD:
            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            label = labels[int(cls_id)]
            if label == 'Aadhar_no':
                region = frame[ymin:ymax, xmin:xmax]
                text = pytesseract.image_to_string(region)
                if len(text.strip()) > 0:
                    aadhar_no = text.strip()
                    break                       
    return aadhar_no

def maskAadhar(img):
    global yolo_model
    detections = yolo_model(img)[0]
    label = None
    for data in detections.boxes.data.tolist():
        confidence = data[4]
        cls_id = data[5]
        if float(confidence) >= CONFIDENCE_THRESHOLD:
            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])            
            label = labels[int(cls_id)]
            if label == 'Aadhar_no':
                max_mask = xmin + int(xmax / 3)
                for y in range(ymin, ymax):
                    for x in range(xmin, max_mask):
                        img[y, x] = [0, 0, 0]    
            cv2.rectangle(img, (xmin, ymin) , (xmax, ymax), GREEN, 2)
            cv2.putText(img, label, (xmin, ymin-10),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)        
    return img

    
#def ViewDoctors(request):
    if request.method == 'GET':
        output = '''
        <div class="admin-content">
            <table class="doctors-table">
                <thead>
                    <tr>
                        <th>Username</th>
                        <th>Phone No</th>
                        <th>Email ID</th>
                        <th>Address</th>
                        <th>Description</th>
                    </tr>
                </thead>
                <tbody>
        '''
        
        mysqlConnect = pymysql.connect(
            host='127.0.0.1',
            port=3306,
            user='root',
            password='root',
            database='redact',
            charset='utf8'
        )
        
        with mysqlConnect:
            result = mysqlConnect.cursor()
            result.execute("select * from user_signup where usertype='Doctor'")
            doctors = result.fetchall()
            
            for doctor in doctors:
                output += f'''
                <tr>
                    <td>{doctor[0]}</td>
                    <td>{doctor[1]}</td>
                    <td>{doctor[2]}</td>
                    <td>{doctor[3]}</td>
                    <td>{doctor[4]}</td>
                </tr>
                '''
        
        output += '''
                </tbody>
            </table>
        </div>
        <style>
            .admin-content {
                background: white;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
                margin-top: 20px;
                overflow-x: auto;
            }
            
            .doctors-table {
                width: 100%;
                border-collapse: collapse;
                font-family: 'Roboto', sans-serif;
            }
            
            .doctors-table th {
                background-color: #2c7be5;
                color: white;
                padding: 12px 15px;
                text-align: left;
                font-weight: 500;
            }
            
            .doctors-table td {
                padding: 12px 15px;
                border-bottom: 1px solid #e0e0e0;
                color: #333;
            }
            
            .doctors-table tr:hover {
                background-color: #f5f9ff;
            }
            
            .doctors-table tr:nth-child(even) {
                background-color: #f8f9fa;
            }
            
            .doctors-table tr:nth-child(even):hover {
                background-color: #f5f9ff;
            }
        </style>
        '''
        
        context = {'data': output}        
        return render(request, 'AdminScreen.html', context)
def ViewDoctors(request):
    if request.method == 'GET':
        # Database connection
        mysqlConnect = pymysql.connect(
            host='127.0.0.1',
            port=3306,
            user='root',
            password='root',
            database='redact',
            charset='utf8'
        )
        
        with mysqlConnect:
            result = mysqlConnect.cursor()
            
            # Get counts for doctors and appointments
            result.execute("SELECT COUNT(*) FROM user_signup WHERE usertype='Doctor'")
            doctor_count = result.fetchone()[0]
            
            result.execute("SELECT COUNT(*) FROM appointment")
            appointment_count = result.fetchone()[0]
            
            # Get all doctors data
            result.execute("SELECT * FROM user_signup WHERE usertype='Doctor'")
            doctors = result.fetchall()
            
            # Build stats HTML
            stats_html = f'''
            <div class="stats-container">
                <div class="stat-card">
                    <h3>Total Doctors</h3>
                    <p>{doctor_count}</p>
                </div>
                <div class="stat-card">
                    <h3>Total Appointments</h3>
                    <p>{appointment_count}</p>
                </div>
            </div>
            '''
            
            # Build table HTML
            table_html = '''
            <div class="admin-content">
                <table class="doctors-table">
                    <thead>
                        <tr>
                            <th>Username</th>
                            <th>Phone No</th>
                            <th>Email ID</th>
                            <th>Address</th>
                            <th>Specialization</th>
                        </tr>
                    </thead>
                    <tbody>
            '''
            
            for doctor in doctors:
                table_html += f'''
                <tr>
                    <td>{doctor[0]}</td>
                    <td>{doctor[1]}</td>
                    <td>{doctor[2]}</td>
                    <td>{doctor[3]}</td>
                    <td>{doctor[4]}</td>
                </tr>
                '''
            
            table_html += '''
                    </tbody>
                </table>
            </div>
            '''
            
            # CSS Styling
            style = '''
            <style>
                /* Stats container styling */
                .stats-container {
                display: flex;
                justify-content: space-between;
                margin: 20px;
                gap: 20px;
            }
            
            .stat-card {
                flex: 1;
                background: white;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                text-align: center;
                transition: transform 0.3s ease;
            }
            
            .stat-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }
            
            .stat-card h3 {
                margin: 0 0 10px 0;
                color: #555;
                font-size: 1.1rem;
            }
            
            .stat-card p {
                margin: 0;
                font-size: 2rem;
                font-weight: bold;
                color: #2196F3;
            }
                
                /* Table styling */
                .admin-content {
                    background: white;
                    border-radius: 8px;
                    padding: 20px;
                    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
                    margin: 20px;
                    overflow-x: auto;
                }
                
                .doctors-table {
                    width: 100%;
                    border-collapse: collapse;
                    font-family: 'Roboto', sans-serif;
                }
                
                .doctors-table th {
                    background-color: #2c7be5;
                    color: white;
                    padding: 12px 15px;
                    text-align: left;
                    font-weight: 500;
                }
                
                .doctors-table td {
                    padding: 12px 15px;
                    border-bottom: 1px solid #e0e0e0;
                    color: #333;
                }
                
                .doctors-table tr:hover {
                    background-color: #f5f9ff;
                }
                
                .doctors-table tr:nth-child(even) {
                    background-color: #f8f9fa;
                }
                
                .doctors-table tr:nth-child(even):hover {
                    background-color: #f5f9ff;
                }
            </style>
            '''
            
            # Combine all components
            context = {'data': style + stats_html + table_html}
            return render(request, 'AdminScreen.html', context)



#def ViewBillingAction(request):
    #if request.method == 'POST':
        global username
        patient_name = request.POST.get('t1', False)
        output = '<table border=1 align=center>'
        output+='<tr><th><font size=3 color=black>Billing ID</font></th>'
        output+='<th><font size=3 color=black>Patient Name</font></th>'
        output+='<th><font size=3 color=black>Bill Amount</font></th>'
        output+='<th><font size=3 color=black>Bill Date</font></th>'
        output+='<th><font size=3 color=black>Bill Description</font></th></tr>'
        mysqlConnect = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'redact',charset='utf8')
        with mysqlConnect:
            result = mysqlConnect.cursor()
            result.execute("select * from billing where patient_name='"+patient_name+"'")
            lists = result.fetchall()
            for ls in lists:
                output+='<tr><td><font size=3 color=black>'+str(ls[0])+'</font></td>'
                output+='<td><font size=3 color=black>'+ls[1]+'</font></td>'
                output+='<td><font size=3 color=black>'+ls[2]+'</font></td>'
                output+='<td><font size=3 color=black>'+ls[3]+'</font></td>'
                output+='<td><font size=3 color=black>'+ls[4]+'</font></td></tr>'
        context= {'data':output}        
        return render(request,'InsuranceScreen.html', context) 

#def ViewBilling(request):
    #if request.method == 'GET':
        return render(request, 'ViewBilling.html', {}) 

def confirmProfileAction(request):
    if request.method == 'POST':
        global username, disease, aadhar_data, aadhar_name, medical_data, medical_name
        today = str(datetime.now())
        aadhar_no = request.POST.get('t1', False)
        dbconnection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'redact',charset='utf8')
        dbcursor = dbconnection.cursor()
        qry = "INSERT INTO patients VALUES('"+str(username)+"','"+disease+"','"+aadhar_no+"','"+aadhar_name+"','"+medical_name+"','"+str(today)+"')"
        dbcursor.execute(qry)
        dbconnection.commit()
        if dbcursor.rowcount == 1:
            data = "Your profile successfully updated in Database"
            context= {'data':data}
            if os.path.exists("PatientApp/static/reports/"+aadhar_name):
                os.remove("PatientApp/static/reports/"+aadhar_name)
            with open("PatientApp/static/reports/"+aadhar_name, "wb") as file:
                file.write(aadhar_data)
            file.close()
            if os.path.exists("PatientApp/static/reports/"+medical_name):
                os.remove("PatientApp/static/reports/"+medical_name)
            with open("PatientApp/static/reports/"+medical_name, "wb") as file:
                file.write(medical_data)
            file.close()
            return render(request,'PatientScreen.html', context)
        else:
            data = "Error in saving your profile"
            context= {'data':data}
            return render(request,'PatientScreen.html', context)

def CreateProfileAction(request):
    if request.method == 'POST':
        global username, disease, aadhar_data, aadhar_name, medical_data, medical_name
        disease = request.POST.get('t1', False)
        aadhar_data = request.FILES['t2'].read()
        aadhar_name = request.FILES['t2'].name
        medical_data = request.FILES['t3'].read()
        medical_name = request.FILES['t3'].name
        if os.path.exists("PatientApp/static/test.jpg"):
            os.remove("PatientApp/static/test.jpg")
        with open("PatientApp/static/test.jpg", "wb") as file:
            file.write(aadhar_data)
        file.close()
        img = cv2.imread("PatientApp/static/test.jpg")
        img = cv2.resize(img, (500, 500))
        aadhar_no = getAadharNo(img)
        output = '<tr><td><font size="3" color="black">Detected&nbsp;Aadhar&nbsp;No</td><td><input type="text" name="t1" size="25" value="'+aadhar_no+'"/></td></tr>'
        context= {'data1':output}
        return render(request,'ConfirmProfile.html', context)        

def CreateProfile(request):
    if request.method == 'GET':
        return render(request, 'CreateProfile.html', {})  

def AppointmentAction(request):
    if request.method == 'POST':
        global username
        doctor = request.POST.get('t1', False)
        disease = request.POST.get('t2', False)
        date = request.POST.get('t3', False)
        today = str(datetime.now())
        bid = 0
        mysqlConnect = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'redact',charset='utf8')
        with mysqlConnect:
            result = mysqlConnect.cursor()
            result.execute("select max(appointment_id) from appointment")
            lists = result.fetchall()
            for ls in lists:
                bid = ls[0]
        if bid != None:
            bid = bid + 1
        else:
            bid = 1
        dbconnection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'redact',charset='utf8')
        dbcursor = dbconnection.cursor()
        qry = "INSERT INTO appointment(appointment_id,patient_name,doctor_name,disease_details,prescription,appointment_date,booking_date) VALUES('"+str(bid)+"','"+username+"','"+doctor+"','"+disease+"','Pending','"+date+"','"+str(today)+"')"
        dbcursor.execute(qry)
        dbconnection.commit()
        if dbcursor.rowcount == 1:
            data = "Your Appointment Confirmed on "+date
            context= {'data':data}
            return render(request,'PatientScreen.html', context)
        else:
            data = "Error in making appointment"
            context= {'data':data}
            return render(request,'PatientScreen.html', context)     
            

def Appointment(request):
    if request.method == 'GET':
        global doctor
        doctor = request.GET['doctor']
        output = '<tr><td><font size="3" color="black">Doctor</td><td><input type="text" name="t1" size="25" value="'+doctor+'" readonly/></td></tr>'
        context= {'data':output}
        return render(request,'BookAppointment.html', context)

def getDetails(user):
    address = ""
    email = ""
    phone = ""
    mysqlConnect = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'redact',charset='utf8')
    with mysqlConnect:
        result = mysqlConnect.cursor()
        result.execute("select phone_no, email, address from user_signup where username='"+user+"'")
        lists = result.fetchall()
        for ls in lists:
            phone = ls[0]
            email = ls[1]
            address = ls[2]
    return phone, email, address        
    

#def ViewPrescription(request):
    if request.method == 'GET':
        global username
        output = '<table border=1 align=center>'
        output+='<tr><th><font size=3 color=black>Appointment ID</font></th>'
        output+='<th><font size=3 color=black>Patient Name</font></th>'
        output+='<th><font size=3 color=black>Doctor Name</font></th>'
        output+='<th><font size=3 color=black>Disease Details</font></th>'
        output+='<th><font size=3 color=black>Prescription</font></th>'
        output+='<th><font size=3 color=black>Appointment Date</font></th>'
        output+='<th><font size=3 color=black>Booking Date</font></th></tr>'
        mysqlConnect = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'redact',charset='utf8')
        with mysqlConnect:
            result = mysqlConnect.cursor()
            result.execute("select * from appointment where patient_name='"+username+"'")
            lists = result.fetchall()
            for ls in lists:
                output+='<tr><td><font size=3 color=black>'+str(ls[0])+'</font></td>'
                output+='<td><font size=3 color=black>'+ls[1]+'</font></td>'
                output+='<td><font size=3 color=black>'+ls[2]+'</font></td>'
                output+='<td><font size=3 color=black>'+ls[3]+'</font></td>'
                output+='<td><font size=3 color=black>'+ls[4]+'</font></td>'
                output+='<td><font size=3 color=black>'+str(ls[5])+'</font></td>'
                output+='<td><font size=3 color=black>'+ls[6]+'</font></td></tr>'
        context= {'data':output}        
        return render(request,'PatientScreen.html', context) 
def ViewPrescription(request):
    if request.method == 'GET':
        global username
        output = '''
        <div class="prescription-content">
            <table class="prescription-table">
                <thead>
                    <tr>
                        <th>Appointment ID</th>
                        <th>Patient Name</th>
                        <th>Doctor Name</th>
                        <th>Disease Details</th>
                        <th>Prescription</th>
                        <th>Appointment Date</th>
                        <th>Booking Date</th>
                    </tr>
                </thead>
                <tbody>
        '''
        
        mysqlConnect = pymysql.connect(
            host='127.0.0.1',
            port=3306,
            user='root',
            password='root',
            database='redact',
            charset='utf8'
        )
        
        with mysqlConnect:
            result = mysqlConnect.cursor()
            result.execute("select * from appointment where patient_name='"+username+"'")
            appointments = result.fetchall()
            
            for appt in appointments:
                output += f'''
                <tr>
                    <td>{appt[0]}</td>
                    <td>{appt[1]}</td>
                    <td>{appt[2]}</td>
                    <td>{appt[3]}</td>
                    <td>{appt[4]}</td>
                    <td>{appt[5]}</td>
                    <td>{appt[6]}</td>
                </tr>
                '''
        
        output += '''
                </tbody>
            </table>
        </div>
        <style>
            .prescription-content {
                background: white;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
                margin: 20px 0;
                overflow-x: auto;
            }
            
            .prescription-table {
                width: 100%;
                border-collapse: collapse;
                font-family: 'Roboto', sans-serif;
            }
            
            .prescription-table th {
                background-color: #2c7be5;
                color: white;
                padding: 12px 15px;
                text-align: left;
                font-weight: 500;
                font-size: 0.95rem;
            }
            
            .prescription-table td {
                padding: 12px 15px;
                border-bottom: 1px solid #e0e0e0;
                color: #333;
                font-size: 0.9rem;
                vertical-align: top;
            }
            
            .prescription-table tr:hover {
                background-color: #f5f9ff;
            }
            
            .prescription-table tr:nth-child(even) {
                background-color: #f8f9fa;
            }
            
            .prescription-table tr:nth-child(even):hover {
                background-color: #f5f9ff;
            }
            
            /* Special formatting for prescription column */
            .prescription-table td:nth-child(5) {
                white-space: pre-wrap;
                max-width: 300px;
                word-break: break-word;
            }
        </style>
        '''
        
        context = {'data': output}        
        return render(request, 'PatientScreen.html', context)
    
def GeneratePrescription(request):
    if request.method == 'GET':
        global username
        bid = request.GET['pid']
        output = '<tr><td><font size="3" color="black">Appointment&nbsp;ID</td><td><input type="text" name="t1" size="25" value="'+bid+'" readonly/></td></tr>'
        context= {'data':output}
        return render(request,'GeneratePrescription.html', context)

def saveBilling(aid):
    global username
    patient_name = ""
    mysqlConnect = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'redact',charset='utf8')
    with mysqlConnect:
        result = mysqlConnect.cursor()
        result.execute("select patient_name from appointment where appointment_id='"+aid+"'")
        lists = result.fetchall()
        for ls in lists:
            patient_name = ls[0]
            break
    today = str(datetime.now())
    dbconnection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'redact',charset='utf8')
    dbcursor = dbconnection.cursor()
    qry = "INSERT INTO billing VALUES('"+str(aid)+"','"+patient_name+"','1000','"+today+"','Consultation with doctor "+username+"')"
    dbcursor.execute(qry)
    dbconnection.commit()      
    

def GeneratePrescriptionAction(request):
    if request.method == 'POST':
        bid = request.POST.get('t1', False)
        prescription = request.POST.get('t2', False)
        dbconnection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'redact',charset='utf8')
        dbcursor = dbconnection.cursor()
        qry = "update appointment set prescription='"+prescription+"' where appointment_id='"+bid+"'"
        dbcursor.execute(qry)
        dbconnection.commit()
        saveBilling(bid)
        if dbcursor.rowcount == 1:
            data = "Prescription Updated Successfully"
            context= {'data':data}
            return render(request,'DoctorScreen.html', context)
        else:
            data = "Error in adding prescription details"
            context= {'data':data}
            return render(request,'DoctorScreen.html', context)

def DownloadAction(request):
    if request.method == 'GET':
        global accessList, username
        name = request.GET.get('requester', False)
        mysqlConnect = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'redact',charset='utf8')
        with mysqlConnect:
            result = mysqlConnect.cursor()
            result.execute("select medical_img from patients where patient_name='"+name+"'")
            lists = result.fetchall()
            for ls in lists:
                name = ls[0]
                break
        with open("PatientApp/static/reports/"+name, "rb") as file:
            data = file.read()
        file.close()        
        response = HttpResponse(data,content_type='application/force-download')
        response['Content-Disposition'] = 'attachment; filename='+name
        return response

def getImage(name):
    aadhar_no = ""
    mysqlConnect = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'redact',charset='utf8')
    with mysqlConnect:
        result = mysqlConnect.cursor()
        result.execute("select aadhar_img, aadhar_no from patients where patient_name='"+name+"'")
        lists = result.fetchall()
        for ls in lists:
            name = ls[0]
            aadhar_no = ls[1]
            break
    return name, aadhar_no

def AdminPatientView(request):
    if request.method == 'GET':
        # HTML table structure
        output = '''
        <div class="appointments-table">
            <table>
                <thead>
                    <tr>
                        <th>Appointment ID</th>
                        <th>Patient Name</th>
                        <th>Doctor Name</th>
                        <th>Disease Details</th>
                        <th>Prescription</th>
                        <th>Appointment Date</th>
                        <th>Booking Date</th>
                        <th>Aadhar No</th>
                        <th>Aadhar Image</th>
                        <th>Medical History</th>
                    </tr>
                </thead>
                <tbody>
        '''

        # Database connection
        mysqlConnect = pymysql.connect(
            host='127.0.0.1',
            port=3306, 
            user='root',
            password='root', 
            database='redact',
            charset='utf8'
        )
        
        with mysqlConnect:
            result = mysqlConnect.cursor()
            
            # Get counts for patients, doctors, and appointments
            result.execute("SELECT COUNT(*) FROM user_signup WHERE usertype='patient'")
            patient_count = result.fetchone()[0]
            
            result.execute("SELECT COUNT(DISTINCT patient_name) AS patient_count_with_appointment from appointment")
            active_patient_count = result.fetchone()[0]

            # Get appointment data
            result.execute("SELECT * FROM appointment")
            lists = result.fetchall()
            
            for ls in lists:
                img, aadhar_no = getImage(ls[1])  # Assuming getImage() is defined elsewhere
                masked_aadhar = "XXXX XXXX " + aadhar_no[-4:] if aadhar_no else "N/A"
                
                output += f'''
                <tr>
                    <td>{ls[0]}</td>
                    <td>{ls[1]}</td>
                    <td>{ls[2]}</td>
                    <td>{ls[3]}</td>
                    <td class="{'pending' if ls[4] == 'Pending' else 'completed'}">{ls[4]}</td>
                    <td>{ls[5]}</td>
                    <td>{ls[6]}</td>
                    <td>{masked_aadhar}</td>
                    <td>
                        <div class="image-container" onclick="showLightbox(this.querySelector('img').src)">
                            <img src="/static/reports/{img}" class="aadhar-image">
                            <div class="image-overlay">Click to enlarge</div>
                        </div>
                    </td>
                    <td>
                        <button class="download-btn" onclick="showPasswordModal('{ls[1]}')">Download</button>
                    </td>
                </tr>
                '''

        output += '''
                </tbody>
            </table>
        </div>
        '''

        # Stats cards HTML
        stats_html = f'''
        <div class="stats-container">
            <div class="stat-card">
                <h3>Total Patients</h3>
                <p>{patient_count}</p>
            </div>
            <div class="stat-card">
                <h3>Active Patients</h3>
                <p>{active_patient_count }</p>
            </div>
        </div>
        '''

        # Password Modal HTML
        modal_html = '''
        <div id="passwordModal" class="modal">
            <div class="modal-content">
                <span class="close" onclick="closeModal()">&times;</span>
                <h3>Enter Admin Password</h3>
                <input type="password" id="downloadPassword" placeholder="Enter admin password" autocomplete="off">
                <button onclick="verifyPassword()">Verify & Download</button>
                <p id="errorMsg" style="color:red;display:none;">Incorrect password!</p>
            </div>
        </div>
        '''

        # CSS Styling (added stats container styling)
        style = '''
        <style>
            /* Stats container styling */
            .stats-container {
                display: flex;
                justify-content: space-between;
                margin: 20px;
                gap: 20px;
            }
            
            .stat-card {
                flex: 1;
                background: white;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                text-align: center;
                transition: transform 0.3s ease;
            }
            
            .stat-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }
            
            .stat-card h3 {
                margin: 0 0 10px 0;
                color: #555;
                font-size: 1.1rem;
            }
            
            .stat-card p {
                margin: 0;
                font-size: 2rem;
                font-weight: bold;
                color: #2196F3;
            }
            
            /* Table styling */
            .appointments-table {
                margin: 20px;
                overflow-x: auto;
            }
            
            .appointments-table table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
                font-size: 0.95rem;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            }
            
            .appointments-table th, .appointments-table td {
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            
            .appointments-table th {
                background-color: #f8f9fa;
                font-weight: 600;
            }
            
            .appointments-table tr:hover {
                background-color: #f5f5f5;
            }
            
            /* Status badges */
            .pending {
                color: #ff9800;
                font-weight: bold;
            }
            
            .completed {
                color: #4caf50;
                font-weight: bold;
            }
            
            /* Image styling */
            .image-container {
                position: relative;
                cursor: pointer;
                transition: transform 0.3s ease;
                display: inline-block;
            }
            
            .aadhar-image {
                width: 250px;
                height: 250px;
                object-fit: contain;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                transition: all 0.3s ease;
                display: block;
            }
            
            .image-overlay {
                position: absolute;
                bottom: 0;
                left: 0;
                right: 0;
                background: rgba(0, 0, 0, 0.7);
                color: white;
                padding: 8px;
                text-align: center;
                font-size: 0.9rem;
                opacity: 0;
                transition: opacity 0.3s ease;
                border-radius: 0 0 6px 6px;
                pointer-events: none;
            }
            
            .image-container:hover .aadhar-image {
                transform: scale(1.05);
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            }
            
            .image-container:hover .image-overlay {
                opacity: 1;
            }
            
            /* Button styling */
            .download-btn {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 12px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 0.9rem;
                transition: background-color 0.3s;
            }
            
            .download-btn:hover {
                background-color: #0b7dda;
            }
            
            /* Modal styling */
            .modal {
                display: none;
                position: fixed;
                z-index: 10000;
                left: 0;
                top: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0,0,0,0.5);
            }
            
            .modal-content {
                background-color: #fefefe;
                margin: 15% auto;
                padding: 25px;
                border: 1px solid #888;
                width: 350px;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
            
            .close {
                color: #aaa;
                float: right;
                font-size: 28px;
                font-weight: bold;
                cursor: pointer;
            }
            
            .close:hover {
                color: black;
            }
            
            .modal-content h3 {
                margin-top: 0;
                color: #333;
            }
            
            .modal-content input[type="password"] {
                width: 100%;
                padding: 12px;
                margin: 15px 0;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 16px;
            }
            
            .modal-content button {
                background-color: #4CAF50;
                color: white;
                padding: 12px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                width: 100%;
                font-size: 16px;
                transition: background-color 0.3s;
            }
            
            .modal-content button:hover {
                background-color: #45a049;
            }
            
            /* Lightbox styling */
            .lightbox {
                display: none;
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0,0,0,0.8);
                z-index: 9999;
                justify-content: center;
                align-items: center;
            }
            
            .lightbox img {
                max-width: 90%;
                max-height: 90%;
                border-radius: 8px;
                box-shadow: 0 0 20px rgba(255,255,255,0.1);
            }
            
            .close-lightbox {
                position: absolute;
                top: 20px;
                right: 20px;
                color: white;
                font-size: 30px;
                cursor: pointer;
            }
        </style>
        '''

        # JavaScript (unchanged)
        script = '''
        <script>
            let currentRequester = '';
            
            function showPasswordModal(requester) {
                currentRequester = requester;
                document.getElementById('passwordModal').style.display = 'block';
                document.getElementById('downloadPassword').focus();
            }
            
            function closeModal() {
                document.getElementById('passwordModal').style.display = 'none';
                document.getElementById('errorMsg').style.display = 'none';
                document.getElementById('downloadPassword').value = '';
            }
            
            function verifyPassword() {
                const password = document.getElementById('downloadPassword').value;
                const errorMsg = document.getElementById('errorMsg');
                
                if (!password) {
                    errorMsg.textContent = "Please enter a password";
                    errorMsg.style.display = 'block';
                    return;
                }
                
                // In production, you would make an AJAX call to verify the password first
                // For this example, we'll just redirect with the password
                window.location.href = `DownloadAction?requester=${currentRequester}&password=${encodeURIComponent(password)}`;
                closeModal();
            }
            
            function showLightbox(imgSrc) {
                const lightbox = document.createElement('div');
                lightbox.className = 'lightbox';
                lightbox.innerHTML = `
                    <span class="close-lightbox" onclick="this.parentElement.remove()">&times;</span>
                    <img src="${imgSrc}">
                `;
                document.body.appendChild(lightbox);
                lightbox.style.display = 'flex';
                
                // Close when clicking anywhere in the lightbox
                lightbox.onclick = function(e) {
                    if (e.target === lightbox) {
                        lightbox.remove();
                    }
                };
            }
            
            // Close modal when clicking outside of it
            window.onclick = function(event) {
                const modal = document.getElementById('passwordModal');
                if (event.target === modal) {
                    closeModal();
                }
            }
            
            // Handle Enter key in password field
            document.addEventListener('DOMContentLoaded', function() {
                document.getElementById('downloadPassword').addEventListener('keyup', function(event) {
                    if (event.key === 'Enter') {
                        verifyPassword();
                    }
                });
            });
        </script>
        '''

        # Combine all elements with stats at the top
        context = {'data': style + stats_html + modal_html + script + output}
        return render(request, 'AdminScreen.html', context)
#def AdminPatientView(request):
    #if request.method == 'GET':
        # HTML table structure
        output = '''
        <div class="appointments-table">
            <table>
                <thead>
                    <tr>
                        <th>Appointment ID</th>
                        <th>Patient Name</th>
                        <th>Doctor Name</th>
                        <th>Disease Details</th>
                        <th>Prescription</th>
                        <th>Appointment Date</th>
                        <th>Booking Date</th>
                        <th>Aadhar No</th>
                        <th>Aadhar Image</th>
                        <th>Medical History</th>
                    </tr>
                </thead>
                <tbody>
        '''

        # Database connection
        mysqlConnect = pymysql.connect(
            host='127.0.0.1',
            port=3306, 
            user='root',
            password='root', 
            database='redact',
            charset='utf8'
        )
        
        with mysqlConnect:
            result = mysqlConnect.cursor()
            result.execute("SELECT * FROM appointment")
            lists = result.fetchall()
            
            for ls in lists:
                img, aadhar_no = getImage(ls[1])  # Assuming getImage() is defined elsewhere
                masked_aadhar = "XXXX XXXX " + aadhar_no[-4:] if aadhar_no else "N/A"
                
                output += f'''
                <tr>
                    <td>{ls[0]}</td>
                    <td>{ls[1]}</td>
                    <td>{ls[2]}</td>
                    <td>{ls[3]}</td>
                    <td class="{'pending' if ls[4] == 'Pending' else 'completed'}">{ls[4]}</td>
                    <td>{ls[5]}</td>
                    <td>{ls[6]}</td>
                    <td>{masked_aadhar}</td>
                    <td>
                        <div class="image-container" onclick="showLightbox(this.querySelector('img').src)">
                            <img src="/static/reports/{img}" class="aadhar-image">
                            <div class="image-overlay">Click to enlarge</div>
                        </div>
                    </td>
                    <td>
                        <button class="download-btn" onclick="showPasswordModal('{ls[1]}')">Download</button>
                    </td>
                </tr>
                '''

        output += '''
                </tbody>
            </table>
        </div>
        '''

        # Password Modal HTML
        modal_html = '''
        <div id="passwordModal" class="modal">
            <div class="modal-content">
                <span class="close" onclick="closeModal()">&times;</span>
                <h3>Enter Admin Password</h3>
                <input type="password" id="downloadPassword" placeholder="Enter admin password" autocomplete="off">
                <button onclick="verifyPassword()">Verify & Download</button>
                <p id="errorMsg" style="color:red;display:none;">Incorrect password!</p>
            </div>
        </div>
        '''

        # CSS Styling
        style = '''
        <style>
            /* Table styling */
            .appointments-table {
                margin: 20px;
                overflow-x: auto;
            }
            
            .appointments-table table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
                font-size: 0.95rem;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            }
            
            .appointments-table th, .appointments-table td {
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            
            .appointments-table th {
                background-color: #f8f9fa;
                font-weight: 600;
            }
            
            .appointments-table tr:hover {
                background-color: #f5f5f5;
            }
            
            /* Status badges */
            .pending {
                color: #ff9800;
                font-weight: bold;
            }
            
            .completed {
                color: #4caf50;
                font-weight: bold;
            }
            
            /* Image styling */
            .image-container {
                position: relative;
                cursor: pointer;
                transition: transform 0.3s ease;
                display: inline-block;
            }
            
            .aadhar-image {
                width: 250px;
                height: 250px;
                object-fit: contain;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                transition: all 0.3s ease;
                display: block;
            }
            
            .image-overlay {
                position: absolute;
                bottom: 0;
                left: 0;
                right: 0;
                background: rgba(0, 0, 0, 0.7);
                color: white;
                padding: 8px;
                text-align: center;
                font-size: 0.9rem;
                opacity: 0;
                transition: opacity 0.3s ease;
                border-radius: 0 0 6px 6px;
                pointer-events: none;
            }
            
            .image-container:hover .aadhar-image {
                transform: scale(1.05);
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            }
            
            .image-container:hover .image-overlay {
                opacity: 1;
            }
            
            /* Button styling */
            .download-btn {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 12px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 0.9rem;
                transition: background-color 0.3s;
            }
            
            .download-btn:hover {
                background-color: #0b7dda;
            }
            
            /* Modal styling */
            .modal {
                display: none;
                position: fixed;
                z-index: 10000;
                left: 0;
                top: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0,0,0,0.5);
            }
            
            .modal-content {
                background-color: #fefefe;
                margin: 15% auto;
                padding: 25px;
                border: 1px solid #888;
                width: 350px;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
            
            .close {
                color: #aaa;
                float: right;
                font-size: 28px;
                font-weight: bold;
                cursor: pointer;
            }
            
            .close:hover {
                color: black;
            }
            
            .modal-content h3 {
                margin-top: 0;
                color: #333;
            }
            
            .modal-content input[type="password"] {
                width: 100%;
                padding: 12px;
                margin: 15px 0;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 16px;
            }
            
            .modal-content button {
                background-color: #4CAF50;
                color: white;
                padding: 12px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                width: 100%;
                font-size: 16px;
                transition: background-color 0.3s;
            }
            
            .modal-content button:hover {
                background-color: #45a049;
            }
            
            /* Lightbox styling */
            .lightbox {
                display: none;
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0,0,0,0.8);
                z-index: 9999;
                justify-content: center;
                align-items: center;
            }
            
            .lightbox img {
                max-width: 90%;
                max-height: 90%;
                border-radius: 8px;
                box-shadow: 0 0 20px rgba(255,255,255,0.1);
            }
            
            .close-lightbox {
                position: absolute;
                top: 20px;
                right: 20px;
                color: white;
                font-size: 30px;
                cursor: pointer;
            }
        </style>
        '''

        # JavaScript
        script = '''
        <script>
            let currentRequester = '';
            
            function showPasswordModal(requester) {
                currentRequester = requester;
                document.getElementById('passwordModal').style.display = 'block';
                document.getElementById('downloadPassword').focus();
            }
            
            function closeModal() {
                document.getElementById('passwordModal').style.display = 'none';
                document.getElementById('errorMsg').style.display = 'none';
                document.getElementById('downloadPassword').value = '';
            }
            
            function verifyPassword() {
                const password = document.getElementById('downloadPassword').value;
                const errorMsg = document.getElementById('errorMsg');
                
                if (!password) {
                    errorMsg.textContent = "Please enter a password";
                    errorMsg.style.display = 'block';
                    return;
                }
                
                // In production, you would make an AJAX call to verify the password first
                // For this example, we'll just redirect with the password
                window.location.href = `DownloadAction?requester=${currentRequester}&password=${encodeURIComponent(password)}`;
                closeModal();
            }
            
            function showLightbox(imgSrc) {
                const lightbox = document.createElement('div');
                lightbox.className = 'lightbox';
                lightbox.innerHTML = `
                    <span class="close-lightbox" onclick="this.parentElement.remove()">&times;</span>
                    <img src="${imgSrc}">
                `;
                document.body.appendChild(lightbox);
                lightbox.style.display = 'flex';
                
                // Close when clicking anywhere in the lightbox
                lightbox.onclick = function(e) {
                    if (e.target === lightbox) {
                        lightbox.remove();
                    }
                };
            }
            
            // Close modal when clicking outside of it
            window.onclick = function(event) {
                const modal = document.getElementById('passwordModal');
                if (event.target === modal) {
                    closeModal();
                }
            }
            
            // Handle Enter key in password field
            document.addEventListener('DOMContentLoaded', function() {
                document.getElementById('downloadPassword').addEventListener('keyup', function(event) {
                    if (event.key === 'Enter') {
                        verifyPassword();
                    }
                });
            });
        </script>
        '''

        context = {'data': style + modal_html + script + output}
        return render(request, 'AdminScreen.html', context)


def DownloadAction(request):
    if request.method == 'GET':
        # Get parameters
        requester = request.GET.get('requester', '')
        password = request.GET.get('password', '')
        
        # Verify password (in production, check against database)
        if password != 'admin123':  # Replace with secure password verification
            return HttpResponse("Unauthorized: Invalid password", status=401)
        
        # Database connection
        mysqlConnect = pymysql.connect(
            host='127.0.0.1',
            port=3306,
            user='root',
            password='root',
            database='redact',
            charset='utf8'
        )
        
        try:
            with mysqlConnect:
                result = mysqlConnect.cursor()
                result.execute("SELECT medical_img FROM patients WHERE patient_name=%s", (requester,))
                lists = result.fetchall()
                
                if not lists or not lists[0][0]:
                    return HttpResponse("Patient record not found", status=404)
                    
                filename = lists[0][0]
                file_path = os.path.join("PatientApp/static/reports/", filename)
                
                if not os.path.exists(file_path):
                    return HttpResponse("Medical file not found", status=404)
                
                with open(file_path, "rb") as file:
                    data = file.read()
                
                response = HttpResponse(data, content_type='application/force-download')
                response['Content-Disposition'] = f'attachment; filename="{filename}"'
                return response
                
        except Exception as e:
            return HttpResponse(f"Error occurred: {str(e)}", status=500)


def DownloadAction1(request):
    print("doctore view")
    if request.method == 'GET':
        # Get parameters
        requester = request.GET.get('requester', '')
        aadhar= requester.split('_')[1]
        patient1=requester.split('_')[0]
        #print("krishna")
        password = request.GET.get('password', '')
        
        # Verify password (in production, check against database)
        if password !=aadhar :  # Replace with secure password verification
            return HttpResponse("Unauthorized: Invalid password", status=401)
        
        # Database connection
        mysqlConnect = pymysql.connect(
            host='127.0.0.1',
            port=3306,
            user='root',
            password='root',
            database='redact',
            charset='utf8'
        )
        
        try:
            with mysqlConnect:
                result = mysqlConnect.cursor()
                result.execute("SELECT medical_img FROM patients WHERE patient_name=%s", (patient1,))
                lists = result.fetchall()
                
                if not lists or not lists[0][0]:
                    return HttpResponse("Patient record not found", status=404)
                    
                filename = lists[0][0]
                file_path = os.path.join("PatientApp/static/reports/", filename)
                
                if not os.path.exists(file_path):
                    return HttpResponse("Medical file not found", status=404)
                
                with open(file_path, "rb") as file:
                    data = file.read()
                
                response = HttpResponse(data, content_type='application/force-download')
                response['Content-Disposition'] = f'attachment; filename="{filename}"'
                return response
                
        except Exception as e:
            return HttpResponse(f"Error occurred: {str(e)}", status=500)

def ViewAppointments(request):
    if request.method == 'GET':
        global username
        today = datetime.now()
        date = today.strftime('%Y-%m-%d')

        mysqlConnect = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='root', 
                                     database='redact', charset='utf8')
        with mysqlConnect:
            result = mysqlConnect.cursor()
            
            # Get counts for stats cards
            result.execute("""
                SELECT 
                    SUM(CASE WHEN appointment_date > %s THEN 1 ELSE 0 END) as upcoming,
                    SUM(CASE WHEN appointment_date = %s THEN 1 ELSE 0 END) as today,
                    SUM(CASE WHEN appointment_date = %s AND prescription = 'Pending' THEN 1 ELSE 0 END) AS active,
                    SUM(CASE WHEN prescription != 'Pending' THEN 1 ELSE 0 END) as generated
                FROM appointment 
                WHERE doctor_name = %s
            """, (date, date, date, username))
            counts = result.fetchone()
            upcoming_count = counts[0] or 0
            today_count = counts[1] or 0
            active_count = counts[2] or 0
            generated_count = counts[3] or 0
            
            # Get appointment data
            result.execute("""
                SELECT * FROM appointment 
                WHERE doctor_name=%s AND appointment_date >= %s 
                ORDER BY appointment_date
            """, (username, date))
            lists = result.fetchall()

            # Build stats HTML
            stats_html = f'''
            <div class="stats-container">
                <div class="stat-card upcoming">
                    <div class="stat-icon">📅</div>
                    <div class="stat-content">
                        <h3>Upcoming</h3>
                        <p>{upcoming_count}</p>
                    </div>
                </div>
                <div class="stat-card today">
                    <div class="stat-icon">📆</div>
                    <div class="stat-content">
                        <h3>Today</h3>
                        <p>{today_count}</p>
                    </div>
                </div>
                <div class="stat-card active">
                    <div class="stat-icon">👨‍⚕️</div>
                    <div class="stat-content">
                        <h3>Active</h3>
                        <p>{active_count}</p>
                    </div>
                </div>
                <div class="stat-card generated">
                    <div class="stat-icon">💊</div>
                    <div class="stat-content">
                        <h3>Generated</h3>
                        <p>{generated_count}</p>
                    </div>
                </div>
            </div>
            '''

            # Build table HTML
            output = '''
            <div class="appointments-table">
                <table>
                    <thead>
                        <tr>
                            <th>Appointment ID</th>
                            <th>Patient Name</th>
                            <th>Doctor Name</th>
                            <th>Disease Details</th>
                            <th>Prescription</th>
                            <th>Appointment Date</th>
                            <th>Booking Date</th>
                            <th>Aadhar No</th>
                            <th>Aadhar Image</th>
                            <th>Download Medical History</th>
                            <th>Generate Description</th>
                        </tr>
                    </thead>
                    <tbody>
            '''

            for ls in lists:
                img, aadhar_no = getImage(ls[1])
                image_path = f"PatientApp/static/reports/{img}"
                processed_image_path = f"PatientApp/static/reports/{ls[1]}_masked.jpg"

                if os.path.exists(image_path):
                    image = cv2.imread(image_path)
                    if image is not None:
                        image = cv2.resize(image, (500, 500))
                        image = maskAadhar(image)
                        cv2.imwrite(processed_image_path, image)

                aadhar_no = "XXXX XXXX " + aadhar_no[-4:]

                output += f'''
                <tr>
                    <td>{ls[0]}</td>
                    <td>{ls[1]}</td>
                    <td>{ls[2]}</td>
                    <td>{ls[3]}</td>
                    <td class="{'pending' if ls[4] == 'Pending' else 'completed'}">{ls[4]}</td>
                    <td>{ls[5]}</td>
                    <td>{ls[6]}</td>
                    <td>{aadhar_no}</td>
                    <td>
                        <div class="image-container" onclick="showLightbox('/static/reports/{ls[1]}_masked.jpg')">
                            <img src="/static/reports/{ls[1]}_masked.jpg" class="aadhar-image">
                            <div class="image-overlay">Click to enlarge</div>
                        </div>
                    </td>
                    <td>
                        <button class="download-btn" onclick="showPasswordModal1('{ls[1]+"_"+ aadhar_no[-4:]}')">Download</button>
                    </td>
                    <td>
                '''

                if ls[4] == 'Pending':
                    output += f'<a href="GeneratePrescription?pid={ls[0]}" class="prescription-btn">Generate Prescription</a>'
                else:
                    output += '<span class="completed-text">Already Generated</span>'

                output += '</td></tr>'

            output += '''
                    </tbody>
                </table>
            </div>
            '''

        # CSS for stats cards
        stats_css = '''
        <style>
            .stats-container {
                display: flex;
                gap: 12px;
                margin-bottom: 20px;
                flex-wrap: wrap;
                justify-content: space-between;
            }
            
            .stat-card {
                flex: 1 1 120px;
                min-width: 100px;
                max-width: 150px;
                background: white;
                border-radius: 8px;
                padding: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .stat-icon {
                font-size: 1.2rem;
                width: 32px;
                height: 32px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
            }

            .stat-card.upcoming .stat-icon {
                background-color: #fff3e0;
                color: #ffa000;
            }

            .stat-card.today .stat-icon {
                background-color: #e3f2fd;
                color: #1976d2;
            }
            
            .stat-card.active .stat-icon {
                background-color: #e8f5e9;
                color: #388e3c;
            }
            
            .stat-card.generated .stat-icon {
                background-color: #f3e5f5;
                color: #8e24aa;
            }
            
            .stat-content h3 {
                margin: 0;
                font-size: 0.8rem;
                color: #666;
                font-weight: 500;
            }
            
            .stat-content p {
                margin: 5px 0 0;
                font-size: 1.2rem;
                font-weight: 600;
                color: #333;
            }
        </style>
        '''

        # Existing CSS
        style1 = '''
        <style>
            .appointments-table {
                margin-top: 30px;
                overflow-x: auto;
            }
            
            .appointments-table table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
                font-size: 0.95rem;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            }
            
            .appointments-table th {
                background-color: #f8f9fa;
                padding: 14px 16px;
                text-align: left;
                color: #4a5568;
                font-weight: 600;
                border-bottom: 2px solid #e0e0e0;
            }
            
            .appointments-table td {
                padding: 12px 16px;
                color: #4a5568;
                border-bottom: 1px solid #e0e0e0;
            }
            
            .appointments-table tr:nth-child(even) {
                background-color: #f9fbfd;
            }
            
            .appointments-table tr:hover {
                background-color: #f0f7ff;
            }
            
            /* Image styling with hover effects */
            .image-container {
                position: relative;
                cursor: pointer;
                display: inline-block;
                transition: transform 0.3s;
            }
            
            .aadhar-image {
                width: 150px;
                height: 150px;
                object-fit: cover;
                border-radius: 4px;
                border: 1px solid #e0e0e0;
                transition: all 0.3s;
            }
            
            .image-overlay {
                position: absolute;
                bottom: 0;
                left: 0;
                right: 0;
                background: rgba(0, 0, 0, 0.7);
                color: white;
                padding: 8px;
                text-align: center;
                font-size: 0.8rem;
                opacity: 0;
                transition: opacity 0.3s;
                border-radius: 0 0 4px 4px;
            }
            
            .image-container:hover .aadhar-image {
                transform: scale(1.05);
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
            
            .image-container:hover .image-overlay {
                opacity: 1;
            }
            
            /* Lightbox styling */
            .lightbox {
                display: none;
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0,0,0,0.9);
                z-index: 9999;
                justify-content: center;
                align-items: center;
            }
            
            .lightbox-content {
                max-width: 90%;
                max-height: 90%;
            }
            
            .lightbox-content img {
                max-width: 100%;
                max-height: 90vh;
                border-radius: 8px;
                box-shadow: 0 0 20px rgba(255,255,255,0.1);
            }
            
            .close-lightbox {
                position: absolute;
                top: 20px;
                right: 20px;
                color: white;
                font-size: 30px;
                cursor: pointer;
            }
            
            .download-btn, .prescription-btn {
                display: inline-block;
                padding: 8px 16px;
                background-color: #2c7be5;
                color: white;
                text-decoration: none;
                border-radius: 4px;
                font-weight: 500;
                transition: all 0.2s;
                font-size: 0.9rem;
            }
            
            .download-btn:hover, .prescription-btn:hover {
                background-color: #1a68d1;
                transform: translateY(-1px);
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            
            .pending {
                color: #d97706;
                font-weight: 500;
            }
            
            .completed {
                color: #059669;
                font-weight: 500;
            }
            
            .completed-text {
                color: #6b7280;
                font-style: italic;
            }

            /* Modal styling */
            .modal {
                display: none;
                position: fixed;
                z-index: 10000;
                left: 0;
                top: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0,0,0,0.5);
            }
            
            .modal-content {
                background-color: #fefefe;
                margin: 15% auto;
                padding: 25px;
                border: 1px solid #888;
                width: 350px;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
            
            .close {
                color: #aaa;
                float: right;
                font-size: 28px;
                font-weight: bold;
                cursor: pointer;
            }
            
            .close:hover {
                color: black;
            }
            
            .modal-content h3 {
                margin-top: 0;
                color: #333;
            }
            
            .modal-content input[type="password"] {
                width: 100%;
                padding: 12px;
                margin: 15px 0;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 16px;
            }
            
            .modal-content button {
                background-color: #4CAF50;
                color: white;
                padding: 12px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                width: 100%;
                font-size: 16px;
                transition: background-color 0.3s;
            }
            
            .modal-content button:hover {
                background-color: #45a049;
            }
        </style>
        '''
        
        # Modal HTML
        modal_html1 = '''
        <div id="passwordModal1" class="modal">
            <div class="modal-content">
                <span class="close" onclick="closeModal()">&times;</span>
                <h3>Enter Admin Password</h3>
                <input type="password" id="downloadPassword1" placeholder="Enter  password" autocomplete="off">
                <button onclick="verifyPassword1()">Verify & Download</button>
                <p id="errorMsg" style="color:red;display:none;">Incorrect password!</p>
            </div>
        </div>
        '''
        
        # JavaScript
        script1 = '''
        <script>
            let currentRequester = '';
            
            function showPasswordModal1(requester) {
                currentRequester = requester;
                document.getElementById('passwordModal1').style.display = 'block';
                document.getElementById('downloadPassword1').focus();
            }
            
            function closeModal() {
                document.getElementById('passwordModal1').style.display = 'none';
                document.getElementById('errorMsg').style.display = 'none';
                document.getElementById('downloadPassword1').value = '';
            }
            
            function verifyPassword1() {
                const password = document.getElementById('downloadPassword1').value;
                const errorMsg = document.getElementById('errorMsg');
                
                if (!password) {
                    errorMsg.textContent = "Please enter a password";
                    errorMsg.style.display = 'block';
                    return;
                }
                
                window.location.href = `DownloadAction1?requester=${currentRequester}&password=${encodeURIComponent(password)}`;
                closeModal();
            }
            
            function showLightbox(imgSrc) {
                const lightbox = document.createElement('div');
                lightbox.className = 'lightbox';
                lightbox.innerHTML = `
                    <span class="close-lightbox" onclick="this.parentElement.remove()">&times;</span>
                    <div class="lightbox-content">
                        <img src="${imgSrc}">
                    </div>
                `;
                document.body.appendChild(lightbox);
                lightbox.style.display = 'flex';
                
                lightbox.onclick = function(e) {
                    if (e.target === lightbox) {
                        lightbox.remove();
                    }
                };
            }
        </script>
        '''

        # Combine all components
        context = {
            'data': stats_css + style1 + stats_html + modal_html1 + script1 + output
        }
        return render(request, 'DoctorScreen.html', context)
#def ViewAppointments(request):
    #if request.method == 'GET':
        #global username
        today = datetime.now()
        date = today.strftime('%Y-%m-%d')

        # Start building the styled table output
        output = '''
        <div class="appointments-table">
            <table>
                <thead>
                    <tr>
                        <th>Appointment ID</th>
                        <th>Patient Name</th>
                        <th>Doctor Name</th>
                        <th>Disease Details</th>
                        <th>Prescription</th>
                        <th>Appointment Date</th>
                        <th>Booking Date</th>
                        <th>Aadhar No</th>
                        <th>Aadhar Image</th>
                        <th>Download Medical History</th>
                        <th>Generate Description</th>
                    </tr>
                </thead>
                <tbody>
        '''

        mysqlConnect = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='root', 
                                     database='redact', charset='utf8')
        with mysqlConnect:
            result = mysqlConnect.cursor()
            result.execute("SELECT * FROM appointment WHERE doctor_name=%s AND appointment_date >= %s ORDER BY appointment_date", (username, date))
            lists = result.fetchall()

            for ls in lists:
                img, aadhar_no = getImage(ls[1])
                image_path = f"PatientApp/static/reports/{img}"
                processed_image_path = f"PatientApp/static/reports/{ls[1]}_masked.jpg"

                if os.path.exists(image_path):
                    image = cv2.imread(image_path)
                    if image is not None:
                        image = cv2.resize(image, (500, 500))
                        image = maskAadhar(image)
                        cv2.imwrite(processed_image_path, image)

                aadhar_no = "XXXX XXXX " + aadhar_no[-4:]

                # Add table row with styled classes
                output += f'''
                <tr>
                    <td>{ls[0]}</td>
                    <td>{ls[1]}</td>
                    <td>{ls[2]}</td>
                    <td>{ls[3]}</td>
                    <td class="{'pending' if ls[4] == 'Pending' else 'completed'}">{ls[4]}</td>
                    <td>{ls[5]}</td>
                    <td>{ls[6]}</td>
                    <td>{aadhar_no}</td>
                    <td>
                        <div class="image-container" onclick="showLightbox('/static/reports/{ls[1]}_masked.jpg')">
                            <img src="/static/reports/{ls[1]}_masked.jpg" class="aadhar-image">
                            <div class="image-overlay">Click to enlarge</div>
                        </div>
                    </td>
                    <td>
                    <!-- <a href="DownloadAction?requester={ls[1]}" class="download-btn">Download</a> -->
                    <button class="download-btn" onclick="showPasswordModal1('{ls[1]+"_"+ aadhar_no[-4:]}')">Download</button>

                    </td>

                    <td>
                '''

                if ls[4] == 'Pending':
                    output += f'<a href="GeneratePrescription?pid={ls[0]}" class="prescription-btn">Generate Prescription</a>'
                else:
                    output += '<span class="completed-text">Already Generated</span>'

                output += '</td></tr>'

        output += '''
                </tbody>
            </table>
        </div>
        '''
# Password Modal HTML
        modal_html1 = '''
        <div id="passwordModal1" class="modal">
            <div class="modal-content">
                <span class="close" onclick="closeModal()">&times;</span>
                <h3>Enter Admin Password</h3>
                <input type="password" id="downloadPassword1" placeholder="Enter admin password" autocomplete="off">
                <button onclick="verifyPassword1()">Verify & Download</button>
                <p id="errorMsg" style="color:red;display:none;">Incorrect password!</p>
            </div>
        </div>
        '''
        # Enhanced CSS with lightbox styling
        style1 = '''
        <style>
            .appointments-table {
                margin-top: 30px;
                overflow-x: auto;
            }
            
            .appointments-table table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
                font-size: 0.95rem;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            }
            
            .appointments-table th {
                background-color: #f8f9fa;
                padding: 14px 16px;
                text-align: left;
                color: #4a5568;
                font-weight: 600;
                border-bottom: 2px solid #e0e0e0;
            }
            
            .appointments-table td {
                padding: 12px 16px;
                color: #4a5568;
                border-bottom: 1px solid #e0e0e0;
            }
            
            .appointments-table tr:nth-child(even) {
                background-color: #f9fbfd;
            }
            
            .appointments-table tr:hover {
                background-color: #f0f7ff;
            }
            
            /* Image styling with hover effects */
            .image-container {
                position: relative;
                cursor: pointer;
                display: inline-block;
                transition: transform 0.3s;
            }
            
            .aadhar-image {
                width: 150px;
                height: 150px;
                object-fit: cover;
                border-radius: 4px;
                border: 1px solid #e0e0e0;
                transition: all 0.3s;
            }
            
            .image-overlay {
                position: absolute;
                bottom: 0;
                left: 0;
                right: 0;
                background: rgba(0, 0, 0, 0.7);
                color: white;
                padding: 8px;
                text-align: center;
                font-size: 0.8rem;
                opacity: 0;
                transition: opacity 0.3s;
                border-radius: 0 0 4px 4px;
            }
            
            .image-container:hover .aadhar-image {
                transform: scale(1.05);
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
            
            .image-container:hover .image-overlay {
                opacity: 1;
            }
            
            /* Lightbox styling */
            .lightbox {
                display: none;
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0,0,0,0.9);
                z-index: 9999;
                justify-content: center;
                align-items: center;
            }
            
            .lightbox-content {
                max-width: 90%;
                max-height: 90%;
            }
            
            .lightbox-content img {
                max-width: 100%;
                max-height: 90vh;
                border-radius: 8px;
                box-shadow: 0 0 20px rgba(255,255,255,0.1);
            }
            
            .close-lightbox {
                position: absolute;
                top: 20px;
                right: 20px;
                color: white;
                font-size: 30px;
                cursor: pointer;
            }
            
            .download-btn, .prescription-btn {
                display: inline-block;
                padding: 8px 16px;
                background-color: #2c7be5;
                color: white;
                text-decoration: none;
                border-radius: 4px;
                font-weight: 500;
                transition: all 0.2s;
                font-size: 0.9rem;
            }
            
            .download-btn:hover, .prescription-btn:hover {
                background-color: #1a68d1;
                transform: translateY(-1px);
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            
            .pending {
                color: #d97706;
                font-weight: 500;
            }
            
            .completed {
                color: #059669;
                font-weight: 500;
            }
            
            .completed-text {
                color: #6b7280;
                font-style: italic;
            }


            /* Button styling */
            .download-btn {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 12px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 0.9rem;
                transition: background-color 0.3s;
            }
            
            .download-btn:hover {
                background-color: #0b7dda;
            }
            
            /* Modal styling */
            .modal {
                display: none;
                position: fixed;
                z-index: 10000;
                left: 0;
                top: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0,0,0,0.5);
            }
            
            .modal-content {
                background-color: #fefefe;
                margin: 15% auto;
                padding: 25px;
                border: 1px solid #888;
                width: 350px;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
            
            .close {
                color: #aaa;
                float: right;
                font-size: 28px;
                font-weight: bold;
                cursor: pointer;
            }
            
            .close:hover {
                color: black;
            }
            
            .modal-content h3 {
                margin-top: 0;
                color: #333;
            }
            
            .modal-content input[type="password"] {
                width: 100%;
                padding: 12px;
                margin: 15px 0;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 16px;
            }
            
            .modal-content button {
                background-color: #4CAF50;
                color: white;
                padding: 12px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                width: 100%;
                font-size: 16px;
                transition: background-color 0.3s;
            }
            
            .modal-content button:hover {
                background-color: #45a049;
            }
            
            
        </style>
        '''
        
        script1 = '''
        <script>

        let currentRequester = '';
            
            function showPasswordModal1(requester) {
                currentRequester = requester;
                document.getElementById('passwordModal1').style.display = 'block';
                document.getElementById('downloadPassword1').focus();
            }
            
            function closeModal() {
                document.getElementById('passwordModal1').style.display = 'none';
                document.getElementById('errorMsg').style.display = 'none';
                document.getElementById('downloadPassword1').value = '';
            }
            
            function verifyPassword1() {
                const password = document.getElementById('downloadPassword1').value;
                const errorMsg = document.getElementById('errorMsg');
                
                if (!password) {
                    errorMsg.textContent = "Please enter a password";
                    errorMsg.style.display = 'block';
                    return;
                }
                
                // In production, you would make an AJAX call to verify the password first
                // For this example, we'll just redirect with the password
                window.location.href = `DownloadAction1?requester=${currentRequester}&password=${encodeURIComponent(password)}`;
                closeModal();
            }
            
            function showLightbox(imgSrc) {
                const lightbox = document.createElement('div');
                lightbox.className = 'lightbox';
                lightbox.innerHTML = `
                    <span class="close-lightbox" onclick="this.parentElement.remove()">&times;</span>
                    <div class="lightbox-content">
                        <img src="${imgSrc}">
                    </div>
                `;
                document.body.appendChild(lightbox);
                lightbox.style.display = 'flex';
                
                // Close when clicking anywhere in the lightbox
                lightbox.onclick = function(e) {
                    if (e.target === lightbox) {
                        lightbox.remove();
                    }
                };
            }
        </script>
        '''

        #context = {'data': style + output}
        context = {'data': style1 + modal_html1 + script1 + output}
        return render(request, 'DoctorScreen.html', context)




def BookAppointment(request):
    if request.method == 'GET':
        output = '''
        <style>
            table {
                width: 70%;
                margin: 30px auto;
                border-collapse: collapse;
                background-color: #f8f9fa;
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
            }
            th {
                background-color: #b3d9ff; /* Light Blue Header */
                color: black;
                padding: 12px;
                text-align: center;
                font-size: 16px;
            }
            td {
                padding: 10px;
                text-align: center;
                color: #333;
                font-size: 16px;
                border-bottom: 1px solid #ddd;
            }
            tr:nth-child(even) {
                background-color: #f2f2f2;
            }
            tr:hover {
                background-color: #e6f2ff; /* Light hover effect */
            }
            .book-btn {
                display: inline-block;
                background-color: #007bff; /* Blue Background */
                color: white;
                padding: 8px 12px;
                border-radius: 5px;
                text-decoration: none;
                font-weight: bold;
                transition: 0.3s ease;
            }
            .book-btn:hover {
                background-color: #0056b3; /* Darker Blue on Hover */
            }
        </style>
        <table align="center">
            <tr>
                <th>Doctor Name</th>
                <th>Specialization</th>
                <th>Address</th>
                <th>Email ID</th>
                <th>Phone No</th>
                
                <th>Book Appointment</th>
            </tr>
        '''
        
        mysqlConnect = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='root', database='redact', charset='utf8')
        with mysqlConnect:
            result = mysqlConnect.cursor()
            result.execute("SELECT username,description , address, email,phone_no FROM user_signup WHERE usertype='Doctor'")
            lists = result.fetchall()
            for ls in lists:
                output += f'''
                <tr>
                    <td>{ls[0]}</td>
                    <td>{ls[1]}</td>
                    <td>{ls[2]}</td>
                    <td>{ls[3]}</td>
                    <td>{ls[4]}</td>
                    <td><a href='Appointment?doctor={ls[0]}' class='book-btn'>Book Now</a></td>
                </tr>
                '''
        
        output += '</table>'
        context = {'data': output}
        return render(request, 'PatientScreen.html', context)





def index(request):
    if request.method == 'GET':
        return render(request,'index.html', {})

def AdminLogin(request):
    if request.method == 'GET':
        return render(request,'AdminLogin.html', {})    

def Register(request):
    if request.method == 'GET':
       return render(request, 'Register.html', {})
    
def DoctorLogin(request):
    if request.method == 'GET':
       return render(request, 'DoctorLogin.html', {})

def PatientLogin(request):
    if request.method == 'GET':
       return render(request, 'PatientLogin.html', {})

#def InsuranceLogin(request):
    #if request.method == 'GET':
       return render(request, 'InsuranceLogin.html', {})
def About(request):
    if request.method == 'GET':
        return render(request, 'about.html', {})    

def isUserExists(username):
    is_user_exists = False
    global details
    mysqlConnect = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'redact',charset='utf8')
    with mysqlConnect:
        result = mysqlConnect.cursor()
        result.execute("select * from user_signup where username='"+username+"'")
        lists = result.fetchall()
        for ls in lists:
            is_user_exists = True
    return is_user_exists    

def RegisterAction(request):
    if request.method == 'POST':
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        contact = request.POST.get('t3', False)
        email = request.POST.get('t4', False)
        address = request.POST.get('t5', False)
        desc = request.POST.get('t6', False)
        usertype = request.POST.get('t7', False)
        record = isUserExists(username)
        page = None
        if record == False:
            dbconnection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'redact',charset='utf8')
            dbcursor = dbconnection.cursor()
            qry = "INSERT INTO user_signup(username,password,phone_no,email,address,description,usertype) VALUES('"+str(username)+"','"+password+"','"+contact+"','"+email+"','"+address+"','"+desc+"','"+usertype+"')"
            dbcursor.execute(qry)
            dbconnection.commit()
            if dbcursor.rowcount == 1:
                data = "Signup Done! You can login now"
                context= {'data':data}
                return render(request,'Register.html', context)
            else:
                data = "Error in signup process"
                context= {'data':data}
                return render(request,'Register.html', context) 
        else:
            data = "Given "+username+" already exists"
            context= {'data':data}
            return render(request,'Register.html', context)


def checkUser(uname, password, utype):
    global username
    msg = "Invalid Login Details"
    mysqlConnect = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'redact',charset='utf8')
    with mysqlConnect:
        result = mysqlConnect.cursor()
        result.execute("select * from user_signup where username='"+uname+"' and password='"+password+"' and usertype='"+utype+"'")
        lists = result.fetchall()
        for ls in lists:
            msg = "success"
            username = uname
            break
    return msg

def PatientLoginAction(request):
    if request.method == 'POST':
        global username
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        msg = checkUser(username, password, "Patient")
        if msg == "success":
            context= {'data':"Welcome "+username}
            return render(request,'PatientScreen.html', context)
        else:
            context= {'data':msg}
            return render(request,'PatientLogin.html', context)
        
def DoctorLoginAction(request):
    if request.method == 'POST':
        global username
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        msg = checkUser(username, password, "Doctor")
        if msg == "success":
            context= {'data':"Welcome "+username}
            return render(request,'DoctorScreen.html', context)
        else:
            context= {'data':msg}
            return render(request,'DoctorLogin.html', context)
        
def AdminLoginAction(request):
    if request.method == 'POST':
        global username
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        if username == "admin" and password == "admin":
            context= {'data':"Welcome "+username}
            return render(request,'AdminScreen.html', context)
        else:
            context= {'data':msg}
            return render(request,'AdminLogin.html', context)

def DoctorLoginAction(request):
    if request.method == 'POST':
        username = request.POST.get('t1', '')
        password = request.POST.get('t2', '')
        msg = checkUser(username, password, "Doctor")
        
        if msg == "success":
            context = {
                'username': username,
                'welcome_message': True,
                'message': "You have successfully logged in to MediConnect"
            }
            return render(request, 'DoctorScreen.html', context)
        else:
            context = {
                'login_error': True,
                'error_message': msg,
                'data': ''  # Add any additional data you want to display
            }
            return render(request, 'DoctorLogin.html', context)

#def InsuranceLoginAction(request):
    #if request.method == 'POST':
        #global username
        #username = request.POST.get('t1', False)
        #password = request.POST.get('t2', False)
        #msg = checkUser(username, password, "Insurance")
        #if msg == "success":
            context= {'data':"Welcome "+username}
            return render(request,'InsuranceScreen.html', context)
        #else:
            context= {'data':msg}
            return render(request,'InsuranceLogin.html', context)




        


        
