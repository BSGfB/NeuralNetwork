/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   MyLog.cpp
 * Author: sergey
 * 
 * Created on November 22, 2016, 6:26 PM
 */

#include "MyLog.h"

MyLog* MyLog::p_instance = 0;
ofstream MyLog::file("log.txt", std::ios_base::out | std::ios_base::trunc);

MyLog& MyLog::operator=(MyLog&) {

}

MyLog::MyLog() {

}

MyLog::MyLog(const MyLog& orig) {

}

MyLog::~MyLog() {
    file.close();
}


MyLog* MyLog::getInstance() {
    if(!p_instance)           
        p_instance = new MyLog();
    return p_instance;
}

void MyLog::addLog(LOG_TYPE logType, std::string logText) {
    std::string msg;
    switch(logType) {
        case LOG_TYPE::ALERT:
            msg = "[ALERT]";
            break;
        case LOG_TYPE::DEBUG:
            msg = "[DEBUG]";
            break;
        case LOG_TYPE::ERROR:
            msg = "[ERROR]";
            break;
        case LOG_TYPE::INFO:
            msg = "[INFO]";
            break;
        case LOG_TYPE::NOTICE:
            msg = "[NOTICE]";
            break;
        case LOG_TYPE::WARNING:
            msg = "[WARNING]";
            break;
        default:
            msg = "[DEFAULT]";
            break;
    }
    if(file.is_open()) {
        file << msg << " " << logText << std::endl;
        file.flush();
    }
}
