//-------------------------------------------------------------------------
// Zum Übersetzen von SVO-Dateien in 3D-Trajektorien
// Erweiterung des Beispiels "tutorial 6 - object detection"
// Kommandozeile: Trajektorienkonverter [Quelle [Ziel [Frames [Start]]]
//   Quelle: Dateiname (*.svo) / IP-Adresse:Port / Kameraparameter
//   Ziel: Dateiname (*.trj)
// 
// (c) 2023 FHNW / STEREOLABS
//-------------------------------------------------------------------------

///////////////////////////////////////////////////////////////////////////
//
// Enthält Codeteile aus Beispielen von STEREOLABS
// 
// Copyright (c) 2023, STEREOLABS.
//
// All rights reserved.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////

/*********************************************************************************
 ** Für mehr Info die Standardbeispiele anschauen
 *********************************************************************************/

 // Standard includes
#include <iostream>
#include <fstream>

// ZED includes
#include <sl/Camera.hpp>

// Using std and sl namespaces
using namespace std;
using namespace sl;
bool is_playback = false;
bool is_trajectory = false;
string trajectoryname;
ofstream trajectory;

void parseArgs(int argc, char** argv, InitParameters& param, int* nframes, int* sframe);
void writeObjects(Objects objects, ObjectDetectionParameters detection_parameters, int frame_position);

int main(int argc, char** argv) {
    Camera zed;
    InitParameters init_parameters;

    // Kommandozeilenvariablen
    int n_frames = 0;
    int s_frame = 0;

    parseArgs(argc, argv, init_parameters, &n_frames, &s_frame);

    // Lokale Variablen
    int nb_detected = 0;
    int frame_position = 0;
    bool is_continued = true;
    char s[2]; // Für interaktiven Modus

    //---------------------------------------------------
    // ZED Objekt mit Hauptparametern
    //---------------------------------------------------
    init_parameters.depth_mode = DEPTH_MODE::ULTRA; // Höhere Genauigkeit bei grossen Distanzen als Default (PERFORMANCE)
    init_parameters.coordinate_units = UNIT::METER;
    init_parameters.depth_maximum_distance = 40; // 3D auch bei grösseren Distanzen (auf Kosten der Genauigkeit)
    init_parameters.sdk_verbose = true;

    if (is_trajectory) {
        trajectory.open(trajectoryname);
        trajectory << "Frame,Objekt,Objekt ID,Objektlabel,Konfidenz,Trackingstatus,x,y,z,vx,xy,vz,width,height,length" << endl;
    }   trajectory.close();

    // Kamera öffnen
    auto returned_state = zed.open(init_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        cout << "Error " << returned_state << ", exit program.\n";
        return EXIT_FAILURE;
    }
    //---------------------------------------------------
    // Allgemeine Detektionsparameter
    //---------------------------------------------------
    // Define the Objects detection module parameters
    ObjectDetectionParameters detection_parameters;
    // run detection for every Camera grab
    detection_parameters.image_sync = true;
    // track detects object accross time and space
    detection_parameters.enable_tracking = true;
    // compute a binary mask for each object aligned on the left image
    detection_parameters.enable_segmentation = true; // designed to give person pixel mask
    
    // If you want to have object tracking you need to enable positional tracking first
    if (detection_parameters.enable_tracking)
        zed.enablePositionalTracking();

    // Detektionsmodell: Mehrere Objekte verschiedener Typen
    detection_parameters.detection_model = OBJECT_DETECTION_MODEL::MULTI_CLASS_BOX_MEDIUM;;

    //---------------------------------------------------
    // Objekterkennung einschalten und Parameter anwenden
    //---------------------------------------------------
    cout << "Object Detection: Loading Module..." << endl;
    returned_state = zed.enableObjectDetection(detection_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        cout << "Error " << returned_state << ", exit program.\n";
        zed.close();
        return EXIT_FAILURE;
    }
 
    //-----------------------------------------------------
    // Laufzeitparameter der Detektion: Erkennungsparameter
    //-----------------------------------------------------
    // Schwellwert der Konfidenz (zwischen 0 und 100)
     int detection_confidence_od = 20;
    // ObjectDetectionRuntimeParameters detection_parameters_rt();
    ObjectDetectionRuntimeParameters detection_parameters_rt(detection_confidence_od);
    // Objektklassen, die detektiert werden sollen
    detection_parameters_rt.object_class_filter = { OBJECT_CLASS::SPORT, OBJECT_CLASS::PERSON,OBJECT_CLASS::FRUIT_VEGETABLE };

    // Hier könnten noch Körperteilerkennungsmerkmale angegeben werden
    // int body_detection_confidence = 60;
    // BodyTrackingRuntimeParameters body_tracking_parameters_rt(body_detection_confidence);

    // Detektierte Objekte
    Objects objects;

    while (is_continued) {
        if (zed.grab() == ERROR_CODE::SUCCESS) {
            //--------------------------------------------------------------
            // Erkennung der Körper mit den angegebenen Detektionsparametern
            //--------------------------------------------------------------
            zed.retrieveObjects(objects, detection_parameters_rt);

            // Das Objekt detektierter Objekte ist neu
            if (objects.is_new) {
                nb_detected++;
                if (is_playback) {
                    frame_position = zed.getSVOPosition();
                    if (!is_trajectory) cout << "Frame: " << frame_position << endl;
                }
                else {
                    frame_position = nb_detected-1;
                    is_continued = is_continued && (nb_detected < 100);
                }

                // Schreiben
                if (frame_position >= s_frame)
                    writeObjects(objects, detection_parameters, frame_position);
                if (!is_trajectory) {
                    cout << "\n'Enter' zum Fortfahren, q'Enter' zum Verlassen...\n";
                    cin.getline(s, 2);
                    is_continued = (strcmp(&s[0], "q"));
                }
            }
        }
        else {
            is_continued = false;
        }
        if (n_frames != 0 && frame_position >= (n_frames+s_frame-1)) is_continued = false;
    } // while

    zed.close();
    if (is_trajectory) trajectory.close();
    return EXIT_SUCCESS;
}

void writeObjects(Objects objects, ObjectDetectionParameters detection_parameters, int frame_position) {
    unsigned int i;
    ObjectData detected_object, first_object;
    if (is_trajectory) {
        if (!objects.object_list.empty()) {
            trajectory.open(trajectoryname, ofstream::app);

            i = 0;
            for (ObjectData detected_object : objects.object_list) {
                cout << frame_position << "," << i << "," <<
                    detected_object.id << "," << detected_object.label << "," <<
                    detected_object.confidence << "," << detected_object.tracking_state << "," <<
                    detected_object.position[0] << "," << detected_object.position[1] << "," << detected_object.position[2] << "," <<
                    detected_object.velocity[0] << "," << detected_object.velocity[1] << "," << detected_object.velocity[2] << endl;

                trajectory << frame_position << "," << i << "," <<
                    detected_object.id << "," << detected_object.label << "," <<
                    detected_object.confidence << "," << detected_object.tracking_state << "," <<
                    detected_object.position[0] << "," << detected_object.position[1] << "," << detected_object.position[2] << "," <<
                    detected_object.velocity[0] << "," << detected_object.velocity[1] << "," << detected_object.velocity[2] << "," <<
                    detected_object.dimensions[0] << "," << detected_object.dimensions[1] << "," << detected_object.dimensions[2] << endl;
                i++;
            }
            trajectory.close();
        }
    } else {
        cout << setprecision(3);
        cout << objects.object_list.size() << " Object(s) detected\n\n";
        if (!objects.object_list.empty()) {

            i = 0;
            for (ObjectData detected_object: objects.object_list) {

                cout << "Objekt " << i << " Attribute:\n";
                cout << " Label '" << detected_object.label << "' (conf. "
                    << detected_object.confidence << "/100)\n";

                if (detection_parameters.enable_tracking)
                    cout << " Tracking ID: " << detected_object.id << " tracking state: " <<
                    detected_object.tracking_state << " / " << detected_object.action_state << "\n";

                cout << " 3D Position: " << detected_object.position <<
                    " Geschwindigkeit: " << detected_object.velocity << "\n";

                cout << " 3D Dimensionen: " << detected_object.dimensions << "\n";

                if (detected_object.mask.isInit())
                    cout << " 2D mask available\n";

                cout << " Bounding Box 2D \n";
                for (auto it : detected_object.bounding_box_2d)
                    cout << "    " << it << "\n";

                cout << " Bounding Box 3D \n";
                for (auto it : detected_object.bounding_box)
                    cout << "    " << it << "\n";
                cout << "\n";
                i++;
            }
        }
    }
}

void parseArgs(int argc, char** argv, InitParameters& param, int* nframes, int* sframe) {
    if (argc > 1 && string(argv[1]).find("-h") != string::npos) {
        cout << "Trajektorienkonverter [Quelle [Ziel [Anzahl_Frames [Start_Frame]]]" << endl <<
            "   Quelle: Dateiname(*.svo) / IP - Adresse : Port / Kameraparameter" << endl <<
            "   Ziel: Dateiname(*.trj)" << endl;
        exit(0);
    }
    if (argc > 1 && string(argv[1]).find(".svo") != string::npos) {
        // SVO input mode
        param.input.setFromSVOFile(argv[1]);
        is_playback = true;
        cout << "Konversion aus SVO-Datei: " << argv[1] << endl;
    }
    else if (argc > 1 && string(argv[1]).find(".svo") == string::npos) {
        string arg = string(argv[1]);
        unsigned int a, b, c, d, port;
        if (sscanf_s(arg.c_str(), "%u.%u.%u.%u:%d", &a, &b, &c, &d, &port) == 5) {
            // Stream input mode - IP + port
            string ip_adress = to_string(a) + "." + to_string(b) + "." + to_string(c) + "." + to_string(d);
            param.input.setFromStream(sl::String(ip_adress.c_str()), port);
            cout << "[Sample] Using Stream input, IP : " << ip_adress << ", port : " << port << endl;
        }
        else if (sscanf_s(arg.c_str(), "%u.%u.%u.%u", &a, &b, &c, &d) == 4) {
            // Stream input mode - IP only
            param.input.setFromStream(sl::String(argv[1]));
            cout << "[Sample] Using Stream input, IP : " << argv[1] << endl;
        }
        else if (arg.find("HD2K") != string::npos) {
            param.camera_resolution = RESOLUTION::HD2K;
            cout << "Kamera-Auflösung HD2K" << endl;
        }
        else if (arg.find("HD1080") != string::npos) {
            param.camera_resolution = RESOLUTION::HD1080;
            cout << "Kamera-Auflösung HD1080" << endl;
        }
        else if (arg.find("HD720") != string::npos) {
            param.camera_resolution = RESOLUTION::HD720;
            cout << "Kamera-Auflösung HD720" << endl;
        }
        else if (arg.find("VGA") != string::npos) {
            param.camera_resolution = RESOLUTION::VGA;
            cout << "[Sample] Using Camera in resolution VGA" << endl;
        }
    }
    if (argc > 2 && string(argv[2]).find(".trj") != string::npos) {
        // Ausgabe in Trajektorendatei
        is_trajectory = true;
        cout << "Konversion in TRJ-Datei: " << argv[2] << endl;
        trajectoryname = string(argv[2]);
    }
    else if (argc > 2 && string(argv[2]).find(".trj") == string::npos) {
        cout << "Unerkanntes Format: " << argv[2] << endl;
    }
    if (argc > 3) {
        sscanf_s(string(argv[3]).c_str(), "%u", nframes);
        cout << "Anzahl Frames: " << *nframes << endl;
        if (argc > 4) {
            sscanf_s(string(argv[4]).c_str(), "%u", sframe);
            cout << "Startframe: " << *sframe << endl;
        }
    }
    cout << endl;
}
