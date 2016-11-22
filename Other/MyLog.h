/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   MyLog.h
 * Author: sergey
 *
 * Created on November 22, 2016, 6:26 PM
 */

#ifndef MYLOG_H
#define MYLOG_H

#include <string>
#include <fstream>

using std::ofstream;

enum LOG_TYPE {ALERT, ERROR, WARNING, NOTICE, INFO, DEBUG};

class MyLog {
private:
    static MyLog * p_instance;
    static ofstream file;
    
    MyLog();
    MyLog(const MyLog& orig);
    MyLog& operator=(MyLog&);
    
    virtual ~MyLog();
public:
    static MyLog* getInstance();
    static void addLog(LOG_TYPE, std::string);

};

#endif /* MYLOG_H */

