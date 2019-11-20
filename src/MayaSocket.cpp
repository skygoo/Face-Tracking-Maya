#include "MayaSocket.h"

#include <string>

#include "opencv2/core/core.hpp"

#include <iostream>
using namespace std;

void MayaSocket::connect() {
    try {
        internal_socket.connect("10.1.120.133", 8088);
        connected = true;
    } catch (int e) {
        cout << "socket error...." << e <<endl;
        connected = false;
    }
}

MayaSocket::MayaSocket() {
    connect();
}

MayaSocket::~MayaSocket() {

}

std::string transMelCmd(std::string name, Point2f pos) {
    //FULKOD! borde kanske basera skalning p� n�got annat, typ ansiktsbredd ist�llet f�r att bara dela med godtyckligt tal? kanske g�ra det m�jligt att v�lja f�r olika delar av ansiktet?
    // Ja tack!
    //Provar att s�tta vikter i mel-scriptet!
    float modifier = 1.4;
    return "setAttr " + name + ".translateX " + std::to_string(pos.x * modifier) + ";\n setAttr " + name + ".translateY " + std::to_string(-pos.y * modifier) + ";\n";
}

int frm = 0;
bool MayaSocket::send(TrackingData &data) {
    std::string cmd = "";

    //cmd += "currentTime " + std::to_string(frm++) + ";\n";
    //cmd += "currentTime (`currentTime -query` + " + std::to_string(data.timeStep*24) + ");\n";
    cmd += "currentTime (`currentTime -query` + 1);\n";

    cmd += transMelCmd("lob",data.markers[LEFTOUTERBROW]);
    cmd += transMelCmd("lib",data.markers[LEFTINNERBROW]);
    cmd += transMelCmd("rib",data.markers[RIGHTINNERBROW]);
    cmd += transMelCmd("rob",data.markers[RIGHTOUTERBROW]);

    cmd += transMelCmd("lc",data.markers[LEFTCHEEK]);
    cmd += transMelCmd("rc",data.markers[RIGHTCHEEK]);

    cmd += transMelCmd("ln",data.markers[LEFTNOSE]);
    cmd += transMelCmd("rn",data.markers[RIGHTNOSE]);

    cmd += transMelCmd("ul",data.markers[UPPERLIP]);

    cmd += transMelCmd("lm",data.markers[LEFTMOUTH]);
    cmd += transMelCmd("rm",data.markers[RIGHTMOUTH]);
    
    cmd += transMelCmd("ll",data.markers[LOWERLIP]);
    
    cmd += "setKeyframe lob lib rib rob lc rc ln rn lm ul rm ll;\n";
    frm++;
    
    try {
        if(internal_socket.send(cmd) < cmd.length()) {
            std::cout << "full message not sent!" << std::endl;
        }
        return true;
    } catch (int e) {
        connect();
        return false;
    }
    
    return true;
}

bool MayaSocket::isConnected() {
    return connected;
}
