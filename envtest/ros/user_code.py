#!/usr/bin/python3


from pickle import NONE
from utils import AgileCommandMode, AgileCommand
from rl_example import rl_example
from scipy.spatial.transform import Rotation as Rot # for rotation matrix representation
from numpy import linalg as LA

import numpy as np # for vector computation
import cv2 # for displaying RGB image
import rospy # for determining the current time
import matplotlib.pyplot as plt # for visualizing data in real time
import math
from csv import writer
from datetime import datetime

initTime = None


def compute_command_vision_based(state, img):
    ################################################
    # !!! Begin of user code !!!
    # TODO: populate the command message
    ################################################
    print("Computing command vision-based!")
    print(state)
    # print("Image shape: ", img.shape)

    # # display the vision 
    # cv2.imshow("Image window", img)
    # cv2.waitKey(0)
    
    # breakpoint() 

    # Example of SRT command
    command_mode = 0
    command = AgileCommand(command_mode)
    command.t = state.t
    command.rotor_thrusts = [1.0, 1.0, 1.0, 1.0]

    # Example of CTBR command
    command_mode = 1
    command = AgileCommand(command_mode)
    command.t = state.t
    command.collective_thrust = 15.0
    command.bodyrates = [0.0, 0.0, 0.0]

    # Example of LINVEL command (velocity is expressed in world frame)
    command_mode = 2
    command = AgileCommand(command_mode)
    command.t = state.t
    command.velocity = [1.0, 0.0, 0.0]
    command.yawrate = 0.0

    ################################################
    # !!! End of user code !!!
    ################################################

    return command


def compute_command_state_based(state, obstacles, vision, start, rl_policy=None):
    ################################################
    # !!! Begin of user code !!!
    # TODO: populate the command message
    ################################################
    # print("Computing command based on obstacle information!")
    # print(state)
    # print("Obstacles: ", obstacles)
    # the obstacles means 
    # The relative position between the center of the quadrotor and the center of the sphere obstacles.
    # The scale means the size of the obstacles, in radius.
    

    # print("started yet: ", start)
    currentTime = -1 # initialize
    now = rospy.get_rostime()
    global initTime
    global date_string # placeholder

    # storage for L1 related variables
    global v_hat_prev # storage of state predictor value of translational speed
    global omega_hat_prev # storage of state predictor value of rotational speed  
    global v_prev # storage of previous vehicle translational speed
    global omega_prev # storage of previous vehicle rotational speed
    global R_prev # storage of previous rotational matrix
    global u_b_prev # storage of previous baseline control action
    global u_ad_prev # storage of previous L1 control action
    global sigma_m_hat_prev # storage of previous sigma_m_hat
    global sigma_um_hat_prev # storage of previous sigma_um_hat
    global lpf1_prev # storage of lpf1 outputs
    global lpf2_prev # storage of lpf2 outputs

    # breakpoint()
    if start: # only start to counting time if we receive the start command
        if initTime is None:
            initTime = now.secs + now.nsecs/1000000000.0
            date_string = f'{datetime.now():%Y-%m-%d %H:%M:%S%z}'
            # initialize L1 needed variables
            v_hat_prev = state.vel
            omega_hat_prev = state.omega
            v_prev = np.array([0.0,0.0,0.0])
            omega_prev = np.array([0.0,0.0,0.0])
    
            # initialize previous rotational matrix
            qq = state.att
            # transforming the quaternion q to rotation matrix R
            r = Rot.from_quat([qq[1],qq[2],qq[3],qq[0]]) # python's quaternion makes the scalar term the last one
            R_prev = r.as_matrix()
            
            u_b_prev = np.array([0.0,0.0,0.0,0.0])
            u_ad_prev = np.array([0.0,0.0,0.0,0.0])
            sigma_m_hat_prev = np.array([0.0,0.0,0.0,0.0])
            sigma_um_hat_prev = np.array([0.0,0.0])
            lpf1_prev = np.array([0.0,0.0,0.0,0.0])
            lpf2_prev = np.array([0.0,0.0,0.0,0.0])
        else:
            currentTime = now.secs + now.nsecs/1000000000.0 - initTime
            print("current time is", currentTime)

            # begin trajectory computation 
            # parameters
            kg_vehicleMass = 0.752 * 1.5 # add small perturbation to the mass
            J = np.array([[0.0025, 0, 0],
                          [0, 0.0021, 0],
                          [0, 0, 0.0043]])
            Jinv = np.linalg.inv(J)
            GeoCtrl_Kpx = 10.0 # 4.512 
            GeoCtrl_Kpy = 10.0 #4.512   
            GeoCtrl_Kpz = 10.0 
            GeoCtrl_Kvx = 2.5 
            GeoCtrl_Kvy = 2.5 # 0.5
            GeoCtrl_Kvz = 4.5
            GeoCtrl_KRx = 0.128
            GeoCtrl_KRy = 0.086
            GeoCtrl_KRz = 0.02
            GeoCtrl_KOx = 0.03
            GeoCtrl_KOy = 0.03 # 0.073
            GeoCtrl_KOz = 0.004

            GRAVITY_MAGNITUDE = 9.8

            l1enable = 0 # if 0, then L1 is disabled
            As_v = -5.0 # parameter for L1
            As_omega = -5.0 # parameter for L1
            ctoffq1Thrust = 5 # cutoff frequency for thrust channel LPF (rad/s)
            ctoffq1Moment = 1 # cutoff frequency for moment channels LPF1 (rad/s)
            ctoffq2Moment = 1 # cutoff frequency for moment channels LPF2 (rad/s)

            zeros3 = [0.0,0.0,0.0]

            # targetPos.x = radius * sinf(currentRate * netTime)
            # targetPos.y = radius * (1 - cosf(currentRate * netTime))
            # targetPos.z = 1
            targetPos = np.array([2*(1-math.cos(currentTime)), 2*math.sin(currentTime), 1.0 + math.sin(currentTime)])
            # targetPos = np.array([1*currentTime,0.0,1.0])

            # targetVel.x = radius * currentRate * cosf(currentRate * netTime)
            # targetVel.y = radius * currentRate * sinf(currentRate * netTime)
            # targetVel.z = 0
            targetVel = np.array([2*math.sin(currentTime), 2*math.cos(currentTime), math.cos(currentTime)])
            # targetVel = np.array([1.0,0.0,0.0])
    
            # targetAcc.x = -radius * currentRate * currentRate * sinf(currentRate * netTime)
            # targetAcc.y = radius * currentRate * currentRate * cosf(currentRate * netTime)
            # targetAcc.z = 0
            targetAcc = np.array([2*math.cos(currentTime), -2*math.sin(currentTime), -math.sin(currentTime)])
            # targetAcc = np.array([0.0,0.0,0.0])
    
            # targetJerk.x = -radius * powF(currentRate,3) * cosf(currentRate * netTime)
            # targetJerk.y = -radius * powF(currentRate,3) * sinf(currentRate * netTime)
            # targetJerk.z = 0
            targetJerk = np.array([-2*math.sin(currentTime), -2*math.cos(currentTime), -math.cos(currentTime)])
            # targetJerk = np.array([0.0,0.0,0.0])
    
            # targetSnap.x = radius * powF(currentRate,4) * sinf(currentRate * netTime)
            # targetSnap.y = -radius * powF(currentRate,4) * cosf(currentRate * netTime)
            # targetSnap.z = 0
            targetSnap = np.array([-2*math.cos(currentTime), 2*math.sin(currentTime), math.sin(currentTime)])
            # targetSnap = np.array([0.0,0.0,0.0])
    
            zeros2 = [0.0,0.0]
            targetYaw = np.array([1.0,0.0])
            targetYaw_dot = np.array(zeros2)
            targetYaw_ddot = np.array(zeros2)
            # targetYaw = np.array([math.cos(currentTime), math.sin(currentTime)])
            # targetYaw_dot = np.array([-math.sin(currentTime), math.cos(currentTime)])
            # targetYaw_ddot = np.array([-math.cos(currentTime), -math.sin(currentTime)])
    
            # begin geometric control
            # Position Error (ep)
            statePos = state.pos
            r_error = statePos - targetPos
    
            # Velocity Error (ev)
            stateVel = state.vel
            v_error = stateVel - targetVel
    
            target_force = np.array(zeros3)
            target_force[0] = kg_vehicleMass * targetAcc[0] - GeoCtrl_Kpx * r_error[0] - GeoCtrl_Kvx * v_error[0]
            target_force[1] = kg_vehicleMass * targetAcc[1] - GeoCtrl_Kpy * r_error[1] - GeoCtrl_Kvy * v_error[1]
            target_force[2] = kg_vehicleMass * (targetAcc[2] + GRAVITY_MAGNITUDE) - GeoCtrl_Kpz * r_error[2] - GeoCtrl_Kvz * v_error[2]
            # change from - GRAVITY_MAGNITUDE to + GRAVITY_MAGNITUDE for upward z-axis
    
            # Z-Axis [zB]
            qq = state.att
    
            # transforming the quaternion q to rotation matrix R
            r = Rot.from_quat([qq[1],qq[2],qq[3],qq[0]]) # python's quaternion makes the scalar term the last one
            R = r.as_matrix()
            
            # breakpoint()
    
            z_axis = R[:,2]
    
            # target thrust [F] (z-positive)
            target_thrust = np.dot(target_force,z_axis)
    
            # Calculate axis [zB_des] (z-positive)
            z_axis_desired = target_force/np.linalg.norm(target_force)
    
            # [xC_des]
            # x_axis_desired = z_axis_desired x [cos(yaw), sin(yaw), 0]^T
            x_c_des = np.array(zeros3)
            x_c_des[0] = targetYaw[0] 
            x_c_des[1] = targetYaw[1] 
            x_c_des[2] = 0                
    
            x_c_des_dot = np.array(zeros3)
            x_c_des_dot[0] = targetYaw_dot[0] 
            x_c_des_dot[1] = targetYaw_dot[1] 
            x_c_des_dot[2] = 0 
    
            x_c_des_ddot = np.array(zeros3)
            x_c_des_ddot[0] = targetYaw_ddot[0] 
            x_c_des_ddot[1] = targetYaw_ddot[1] 
            x_c_des_ddot[2] = 0  
    
            # [yB_des]
            y_axis_desired = np.cross(z_axis_desired, x_c_des)
            y_axis_desired = y_axis_desired/np.linalg.norm(y_axis_desired)
            # [xB_des]
            x_axis_desired = np.cross(y_axis_desired, z_axis_desired)
    
            # [eR]
            # Slow version
            Rdes = np.empty(shape=(3,3))
            Rdes[:,0] = x_axis_desired
            Rdes[:,1] = y_axis_desired
            Rdes[:,2] = z_axis_desired
            # Matrix3f Rdes(Vector3f(x_axis_desired.x, y_axis_desired.x, z_axis_desired.x),
            #               Vector3f(x_axis_desired.y, y_axis_desired.y, z_axis_desired.y),
            #               Vector3f(x_axis_desired.z, y_axis_desired.z, z_axis_desired.z));
    
            eRM = (np.matmul(Rdes.transpose(),R) - np.matmul(R.transpose(), Rdes)) / 2
    
            # Matrix3<T>(const T ax, const T ay, const T az,
            #  const T bx, const T by, const T bz,
            #  const T cx, const T cy, const T cz)
            # eR.x = eRM.c.y;
            # eR.y = eRM.a.z;
            # eR.z = eRM.b.x;
            eR = np.array(zeros3)
            eR[0] = eRM[2,1]
            eR[1] = eRM[0,2]
            eR[2] = eRM[1,0]
            
            Omega = state.omega
            #     compute Omegad    
            a_error = np.array(zeros3) # error on acceleration       
            a_error = [0,0,-GRAVITY_MAGNITUDE] + R[:,2]* target_thrust / kg_vehicleMass - targetAcc     
            # turn GRAVITY_MAGNITUDE to - GRAVITY_MAGNITUDE       
            # turn - R[:,2]* target_thrust / kg_vehicleMass to + R[:,2]* target_thrust / kg_vehicleMass  

            target_force_dot = np.array(zeros3) # derivative of target_force      
            target_force_dot[0] = - GeoCtrl_Kpx * v_error[0] - GeoCtrl_Kvx * a_error[0] + kg_vehicleMass * targetJerk[0]       
            target_force_dot[1] = - GeoCtrl_Kpy * v_error[1] - GeoCtrl_Kvy * a_error[1] + kg_vehicleMass * targetJerk[1]    
            target_force_dot[2] = - GeoCtrl_Kpz * v_error[2] - GeoCtrl_Kvz * a_error[2] + kg_vehicleMass * targetJerk[2]     
            hatOperatorOmega = hatOperator(Omega)     
            
            b3_dot = np.matmul(np.matmul(R, hatOperatorOmega),[0,0,1])
            target_thrust_dot = + np.dot(target_force_dot,R[:,2]) + np.dot(target_force, b3_dot)
            # turn the RHS from - to +
    
            j_error = np.array(zeros3) # error on jerk
            j_error = np.dot(R[:,2], target_thrust_dot) / kg_vehicleMass + b3_dot * target_thrust / kg_vehicleMass - targetJerk
            # turn - np.dot(R[:,2], target_thrust_dot) / kg_vehicleMass to np.dot(R[:,2], target_thrust_dot) / kg_vehicleMass
            # turn - b3_dot * target_thrust / kg_vehicleMass to + b3_dot * target_thrust / kg_vehicleMass
    
            target_force_ddot = np.array(zeros3) # derivative of target_force_dot
            target_force_ddot[0] = - GeoCtrl_Kpx * a_error[0] - GeoCtrl_Kvx * j_error[0] + kg_vehicleMass * targetSnap[0]
            target_force_ddot[1] = - GeoCtrl_Kpy * a_error[1] - GeoCtrl_Kvy * j_error[1] + kg_vehicleMass * targetSnap[1]
            target_force_ddot[2] = - GeoCtrl_Kpz * a_error[2] - GeoCtrl_Kvz * j_error[2] + kg_vehicleMass * targetSnap[2]
    
            b3cCollection = np.array([zeros3,zeros3,zeros3]) # collection of three three-dimensional vectors b3c, b3c_dot, b3c_ddot
            b3cCollection = unit_vec(-target_force, -target_force_dot, -target_force_ddot) # unit_vec function is from geometric controller's git repo: https://github.com/fdcl-gwu/uav_geometric_control/blob/master/matlab/aux_functions/deriv_unit_vector.m
    
            b3c = np.array(zeros3)
            b3c_dot = np.array(zeros3)
            b3c_ddot = np.array(zeros3)
    
            b3c[0] = b3cCollection[0]
            b3c[1] = b3cCollection[1]
            b3c[2] = b3cCollection[2]
    
            b3c_dot[0] = b3cCollection[3]
            b3c_dot[1] = b3cCollection[4]
            b3c_dot[2] = b3cCollection[5]
    
            b3c_ddot[0] = b3cCollection[6]
            b3c_ddot[1] = b3cCollection[7]
            b3c_ddot[2] = b3cCollection[8]
    
            A2 = - np.matmul(hatOperator(x_c_des), b3c)
            A2_dot = - np.matmul(hatOperator(x_c_des_dot),b3c) - np.matmul(hatOperator(x_c_des), b3c_dot)
            A2_ddot = - np.matmul(hatOperator(x_c_des_ddot), b3c) - np.matmul(hatOperator(x_c_des_dot), b3c_dot) * 2 - np.matmul(hatOperator(x_c_des), b3c_ddot)
    
            b2cCollection = np.array([zeros3,zeros3,zeros3]) # collection of three three-dimensional vectors b2c, b2c_dot, b2c_ddot
            b2cCollection = unit_vec(A2, A2_dot, A2_ddot) # unit_vec function is from geometric controller's git repo: https://github.com/fdcl-gwu/uav_geometric_control/blob/master/matlab/aux_functions/deriv_unit_vector.m
    
            b2c = np.array(zeros3)
            b2c_dot = np.array(zeros3)
            b2c_ddot = np.array(zeros3)
    
            b2c[0] = b2cCollection[0]
            b2c[1] = b2cCollection[1]
            b2c[2] = b2cCollection[2]
    
            b2c_dot[0] = b2cCollection[3]
            b2c_dot[1] = b2cCollection[4]
            b2c_dot[2] = b2cCollection[5]
    
            b2c_ddot[0] = b2cCollection[6]
            b2c_ddot[1] = b2cCollection[7]
            b2c_ddot[2] = b2cCollection[8]
    
            b1c_dot = np.matmul(hatOperator(b2c_dot), b3c) + np.matmul(hatOperator(b2c), b3c_dot)
            b1c_ddot = np.matmul(hatOperator(b2c_ddot),b3c) + np.matmul(hatOperator(b2c_dot), b3c_dot) * 2 + np.matmul(hatOperator(b2c), b3c_ddot)
    
            Rd_dot = np.empty(shape=(3,3)) # derivative of Rdes
            Rd_ddot = np.empty(shape=(3,3)) # derivative of Rd_dot
    
            Rd_dot[0,:] = b1c_dot
            Rd_dot[1,:] = b2c_dot
            Rd_dot[2,:] = b3c_dot
            Rd_dot = Rd_dot.transpose();
    
            Rd_ddot[0,:] = b1c_ddot
            Rd_ddot[1,:] = b2c_ddot
            Rd_ddot[2,:] = b3c_ddot
            Rd_ddot = Rd_ddot.transpose();
    
            Omegad = veeOperator(np.matmul(Rdes.transpose(), Rd_dot))
            Omegad_dot = veeOperator(np.matmul(Rdes.transpose(), Rd_ddot) - np.matmul(hatOperator(Omegad), hatOperator(Omegad)))
            
            # these two lines are remedy which is not supposed to exist in the code. There might be an error in the code above.
            # Omegad[1] = -Omegad[1]
            # Omegad_dot[1] = -Omegad_dot[1]

            # temporarily use zero Omegad
            ew = Omega -  np.matmul(np.matmul(R.transpose(), Rdes), Omegad)
    
            # Moment: simple version
            M = np.array(zeros3)
            M[0] = -GeoCtrl_KRx * eR[0] - GeoCtrl_KOx * ew[0]
            M[1] = -GeoCtrl_KRy * eR[1] - GeoCtrl_KOy * ew[1]
            M[2] = -GeoCtrl_KRz * eR[2] - GeoCtrl_KOz * ew[2]
            # Moment: full version 
            M = M - np.matmul(J, (np.matmul(hatOperator(Omega), np.matmul(R.transpose(),np.matmul(Rdes, Omegad))) - np.matmul(R.transpose(), np.matmul(Rdes, Omegad_dot))))
            # ShengC: an additive term is the following
            momentAdd = np.cross(Omega, (np.matmul(J, Omega))) # J is the inertia matrix
            M = M +  momentAdd

            thrustMomentCmd = np.array([0.0,0.0,0.0,0.0])
            thrustMomentCmd[0] = target_thrust
            thrustMomentCmd[1] = M[0]
            thrustMomentCmd[2] = M[1]
            thrustMomentCmd[3] = M[2]
    

            # == begin L1 adaptive control ==
            # first do the state predictor

            e3 = np.array([0.0, 0.0, 1.0])
            dt = 0.01


            # load translational velocity
            v_now = state.vel
    

            # load rotational velocity
            omega_now = state.omega

            massInverse = 1.0 / kg_vehicleMass

            # compute prediction error (on previous step)
            vpred_error_prev = v_hat_prev - v_prev
            omegapred_error_prev = omega_hat_prev - omega_prev

            # NOTE: per the definition in vector3.h, vector * scalaer must have vector first then multiply the scalar later
            # original form:
            v_hat = v_hat_prev + (-e3 * GRAVITY_MAGNITUDE + R_prev[:,2]* (u_b_prev[0] + u_ad_prev[0] + sigma_m_hat_prev[0]) * massInverse + R_prev[:,0] * sigma_um_hat_prev[0] * massInverse + R_prev[:,1] * sigma_um_hat_prev[1] * massInverse + vpred_error_prev * As_v) * dt
            # turn e3 * GRAVITY_MAGNITUDE to -e3 * GRAVITY_MAGNITUDE
            # turn - R_prev[:,2]* (u_b_prev[0] + u_ad_prev[0] + sigma_m_hat_prev[0]) to + R_prev[:,2]* (u_b_prev[0] + u_ad_prev[0] + sigma_m_hat_prev[0])
            # breakpoint()
            # Jinv is the inverse of inertia J

            # temp vector: thrustMomentCmd[1--3] + u_ad_prev[1--3] + sigma_m_hat_prev[1--3]
            # original form
            tempVec = np.array([u_b_prev[1] + u_ad_prev[1] + sigma_m_hat_prev[1], u_b_prev[2] + u_ad_prev[2] + sigma_m_hat_prev[2], u_b_prev[3] + u_ad_prev[3] + sigma_m_hat_prev[3]])
            omega_hat = omega_hat_prev + (-np.matmul(Jinv, np.cross(omega_prev, (np.matmul(J, omega_prev)))) + np.matmul(Jinv, tempVec) + omegapred_error_prev * As_omega) * dt

            # update the state prediction storage
            v_hat_prev = v_hat
            omega_hat_prev = omega_hat

            # compute prediction error (for this step)
            vpred_error = v_hat - v_now
            omegapred_error = omega_hat - omega_now

            # exponential coefficients coefficient for As
            exp_As_v_dt = math.exp(As_v * dt)
            exp_As_omega_dt = math.exp(As_omega * dt)

            # later part of uncertainty estimation (piecewise constant)
            PhiInvmu_v = vpred_error / (exp_As_v_dt - 1) * As_v * exp_As_v_dt
            PhiInvmu_omega = omegapred_error / (exp_As_omega_dt - 1) * As_omega * exp_As_omega_dt

            sigma_m_hat = np.array([0.0,0.0,0.0,0.0]) # estimated matched uncertainty
            sigma_m_hat_2to4 = np.array([0.0,0.0,0.0]) # second to fourth element of the estimated matched uncertainty
            sigma_um_hat = np.array([0.0,0.0]) # estimated unmatched uncertainty

            # use the rotation matrix in the current step
            qq = state.att
            # transforming the quaternion q to rotation matrix R
            r = Rot.from_quat([qq[1],qq[2],qq[3],qq[0]]) # python's quaternion makes the scalar term the last one
            R = r.as_matrix()


            sigma_m_hat[0] = - np.dot(R[:,2], PhiInvmu_v) * kg_vehicleMass
            # turn np.dot(R[:,2], PhiInvmu_v) * kg_vehicleMass to -np.dot(R[:,2], PhiInvmu_v) * kg_vehicleMass
            sigma_m_hat_2to4 = -np.matmul(J, PhiInvmu_omega)
            sigma_m_hat[1] = sigma_m_hat_2to4[0]
            sigma_m_hat[2] = sigma_m_hat_2to4[1]
            sigma_m_hat[3] = sigma_m_hat_2to4[2]

            sigma_um_hat[0] = -np.dot(R[:,0], PhiInvmu_v) * kg_vehicleMass
            sigma_um_hat[1] = -np.dot(R[:,1], PhiInvmu_v) * kg_vehicleMass

            # store uncertainty estimations
            sigma_m_hat_prev = sigma_m_hat
            sigma_um_hat_prev = sigma_um_hat

            # compute lpf1 coefficients
            lpf1_coefficientThrust1 = math.exp(- ctoffq1Thrust * dt)
            lpf1_coefficientThrust2 = 1.0 - lpf1_coefficientThrust1

            lpf1_coefficientMoment1 = math.exp(- ctoffq1Moment * dt)
            lpf1_coefficientMoment2 = 1.0 - lpf1_coefficientMoment1

            # update the adaptive control
            u_ad_int = np.array([0.0,0.0,0.0,0.0])
            u_ad = np.array([0.0,0.0,0.0,0.0])

            # low-pass filter 1 (negation is added to u_ad_prev to filter the correct signal)
            u_ad_int[0] = lpf1_coefficientThrust1 * (lpf1_prev[0]) + lpf1_coefficientThrust2 * sigma_m_hat[0]
            u_ad_int[1:3] = lpf1_coefficientMoment1 * (lpf1_prev[1:3]) + lpf1_coefficientMoment2 * sigma_m_hat[1:3]
            # u_ad_int[2] = lpf1_coefficientMoment1 * (lpf1_prev[2]) + lpf1_coefficientMoment2 * sigma_m_hat[2];
            # u_ad_int[3] = lpf1_coefficientMoment1 * (lpf1_prev[3]) + lpf1_coefficientMoment2 * sigma_m_hat[3];

            lpf1_prev = u_ad_int; # store the current state

            # coefficients for the second LPF on the moment channel
            lpf2_coefficientMoment1 = math.exp(- ctoffq2Moment * dt)
            lpf2_coefficientMoment2 = 1.0 - lpf2_coefficientMoment1

            # low-pass filter 2 (optional)
            u_ad[0] = u_ad_int[0] # only one filter on the thrust channel
            u_ad[1:3] = lpf2_coefficientMoment1 * lpf2_prev[1:3] + lpf2_coefficientMoment2 * u_ad_int[1:3]
            # u_ad[2] = lpf2_coefficientMoment1 * lpf2_prev[2] + lpf2_coefficientMoment2 * u_ad_int[2];
            # u_ad[3] = lpf2_coefficientMoment1 * lpf2_prev[3] + lpf2_coefficientMoment2 * u_ad_int[3];

            lpf2_prev = u_ad # store the current state
    
            u_ad = -u_ad

            # store the values for next iteration
            u_ad_prev = u_ad * l1enable

            v_prev = v_now
            omega_prev = omega_now
            R_prev = R
            u_b_prev = thrustMomentCmd

            # == end L1 adaptive control ==

            # compute motor command
            individualMotorCmd = np.array([0.0,0.0,0.0,0.0])
            motorAssignMatrix = np.array([[1, 1, 1, 1],
                                      [-0.1, 0.1,-0.1, 0.1],
                                      [-0.075, 0.075, 0.075, -0.075],
                                      [-0.022, -0.022, 0.022, 0.022]])
            individualMotorCmd = np.matmul(np.linalg.inv(motorAssignMatrix),thrustMomentCmd + u_ad_prev)
             


            # == logging ==
            logging_R = r.as_euler('zyx',degrees=True)
            rdes = Rot.from_matrix(Rdes)
            logging_Rdes = rdes.as_euler('zyx',degrees=True)

            with open(date_string+'.csv', 'a', newline='') as f_object:  
                writer_obj = writer(f_object)
                writer_obj.writerow([state.t, 
                                    statePos[0],
                                    statePos[1],
                                    statePos[2],
                                    targetPos[0],
                                    targetPos[1],
                                    targetPos[2],
                                    stateVel[0],
                                    stateVel[1],
                                    stateVel[2],
                                    targetVel[0],
                                    targetVel[1],
                                    targetVel[2],
                                    logging_R[0],
                                    logging_R[1],
                                    logging_R[2], 
                                    logging_Rdes[0],
                                    logging_Rdes[1],
                                    logging_Rdes[2],
                                    Omega[0],
                                    Omega[1],
                                    Omega[2],
                                    Omegad[0],
                                    Omegad[1],
                                    Omegad[2],
                                    v_hat_prev[0],
                                    v_hat_prev[1],
                                    v_hat_prev[2],
                                    omega_hat_prev[0],
                                    omega_hat_prev[1],
                                    omega_hat_prev[2],
                                    sigma_m_hat[0],
                                    sigma_m_hat[1],
                                    sigma_m_hat[2],
                                    sigma_m_hat[3],
                                    sigma_um_hat[0],
                                    sigma_um_hat[1],
                                    u_ad[0],
                                    u_ad[1],
                                    u_ad[2],
                                    u_ad[3]])
            f_object.close()

            # Example of SRT command
            command_mode = 0
            command = AgileCommand(command_mode)
            command.t = state.t
            command.rotor_thrusts = individualMotorCmd
            # first element is for forward right motor

            # command_mode = 1
            # command = AgileCommand(command_mode)
            # command.t = state.t
            # command.collective_thrust = target_thrust/kg_vehicleMass
            # command.bodyrates = [Omegad[0],Omegad[1],Omegad[2]] # note the direction of Omegad has been flipped
            
    else:
        print("waiting for the start")

    # plt.scatter(state.t,individualMotorCmd[0])
    # plt.show()
    # breakpoint()
    # print("geometric control command is", thrustMomentCmd)
    # # end of geometric control

    # # display the vision 
    # cv2.destroyAllWindows()
    # cv2.imshow("Image window", vision)
    # cv2.waitKey(0)

    # breakpoint()  # use this command to set up an break point to inspect variable values

    # fill in the command when geometric computation has not started
    if currentTime < 0:
        command_mode = 0
        command = AgileCommand(command_mode)
        command.t = state.t
        command.rotor_thrusts = [0,0,0,0]

    # # Example of CTBR command
    # command_mode = 1
    # command = AgileCommand(command_mode)
    # command.t = state.t
    # command.collective_thrust = 10.0
    # command.bodyrates = [0.0, 0.0, 0.0]

    # # Example of LINVEL command (velocity is expressed in world frame)
    # command_mode = 2
    # command = AgileCommand(command_mode)
    # command.t = state.t
    # # only send forward command if the drone's x position is within 61 m
    # if state.pos[0] <= 61: 
    #     command.velocity = [3.0, 0.0, 0.0]
    # else:
    #     command.velocity = [0.0, 0.0, 0.0]
    # command.yawrate = 0.0

    # If you want to test your RL policy
    if rl_policy is not None:
        command = rl_example(state, obstacles, rl_policy)

    ################################################
    # !!! End of user code !!!
    ################################################

    return command

def unit_vec(q, q_dot, q_ddot):
    collection = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    nq = LA.norm(q)
    u = q / nq
    u_dot = q_dot / nq - q * np.dot(q, q_dot) / pow(nq,3)

    u_ddot = q_ddot / nq - q_dot / pow(nq,3) * (2 * np.dot(q, q_dot)) - q / pow(nq,3) * (np.dot(q_dot, q_dot) + np.dot(q, q_ddot)) + 3 * q / pow(nq,5) * pow(np.dot(q, q_dot),2)
    # breakpoint()
    collection[0 : 3] = u
    collection[3 : 6] = u_dot
    collection[6 : 9] = u_ddot

    # breakpoint()

    return collection

def hatOperator(v):
    hat = np.zeros((3, 3), dtype=float)
    # breakpoint()
    hat[2,1] = v[0]
    hat[1,2] = -v[0]
    hat[0,2] = v[1]
    hat[2,0] = -v[1]
    hat[1,0] = v[2]
    hat[0,1] = -v[2]
    return hat

def veeOperator(input):
    output = np.zeros((3), dtype=float)
    # const T ax, const T ay, const T az,
    # const T bx, const T by, const T bz,
    # const T cx, const T cy, const T cz
    output[0] = input[2][1]
    output[1] = input[0][2]
    output[2] = input[1][0]

    return output
