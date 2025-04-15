from django.urls import path

from . import views

urlpatterns = [path("index.html", views.index, name="index"),
	       path('DoctorLogin.html', views.DoctorLogin, name="DoctorLogin"), 
	       path('PatientLogin.html', views.PatientLogin, name="PatientLogin"), 
	       path('AdminLogin.html', views.AdminLogin, name="AdminLogin"), 
	       path('AdminLoginAction', views.AdminLoginAction, name="AdminLoginAction"), 
	       path('about/', views.About, name='about'),
	       #path('InsuranceLoginAction', views.InsuranceLoginAction, name="InsuranceLoginAction"), 	
	       path('Register.html', views.Register, name="Register"),
	       path('RegisterAction', views.RegisterAction, name="RegisterAction"),	
	       path('BookAppointment', views.BookAppointment, name="BookAppointment"),
	       path('AppointmentAction', views.AppointmentAction, name="AppointmentAction"),
	       path('Appointment', views.Appointment, name="Appointment"),
	       path('ViewPrescription', views.ViewPrescription, name="ViewPrescription"),
	       path('ViewAppointments', views.ViewAppointments, name="ViewAppointments"),
	       path('GeneratePrescription', views.GeneratePrescription, name="GeneratePrescription"),
	       path('GeneratePrescriptionAction', views.GeneratePrescriptionAction, name="GeneratePrescriptionAction"),
	       path('DoctorLoginAction', views.DoctorLoginAction, name="DoctorLoginAction"), 
	       path('PatientLoginAction', views.PatientLoginAction, name="PatientLoginAction"), 	
	       path('CreateProfile.html', views.CreateProfile, name="CreateProfile"),
	       path('CreateProfileAction', views.CreateProfileAction, name="CreateProfileAction"),	
	       #path('ViewBilling', views.ViewBilling, name="ViewBilling"),
	       #path('ViewBillingAction', views.ViewBillingAction, name="ViewBillingAction"),
	       path('confirmProfileAction', views.confirmProfileAction, name="confirmProfileAction"),
	       path('DownloadAction', views.DownloadAction, name="DownloadAction"),
           path('DownloadAction1', views.DownloadAction1, name="DownloadAction1"),
           
	       path('AdminPatientView', views.AdminPatientView, name="AdminPatientView"),
	       path('ViewDoctors', views.ViewDoctors, name="ViewDoctors"),
]
