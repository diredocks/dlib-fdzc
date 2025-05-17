#
#  _______ _    _ _____  _____     _____  _____                 
# |__   __| |  | |_   _|/ ____|   |_   _|/ ____|       /\       
#    | |  | |__| | | | | (___       | | | (___        /  \      
#    | |  |  __  | | |  \___ \      | |  \___ \      / /\ \     
#    | |  | |  | |_| |_ ____) |    _| |_ ____) |    / ____ \    
#    |_|__|_|_ |_|_____|_____/__  |_____|_____/    /_/  _ \_\   
#   |__   __| |  | |__   __/ __ \|  __ \|_   _|   /\   | |      
#      | |  | |  | |  | | | |  | | |__) | | |    /  \  | |      
#      | |  | |  | |  | | | |  | |  _  /  | |   / /\ \ | |      
#      | |  | |__| |  | | | |__| | | \ \ _| |_ / ____ \| |____  
#      |_|   \____/   |_|  \____/|_|  \_\_____/_/    \_\______| 
#
#
#    _____  ______          _____      _______ _    _ ______       
#   |  __ \|  ____|   /\   |  __ \    |__   __| |  | |  ____|      
#   | |__) | |__     /  \  | |  | |      | |  | |__| | |__         
#   |  _  /|  __|   / /\ \ | |  | |      | |  |  __  |  __|        
#   | | \ \| |____ / ____ \| |__| |      | |  | |  | | |____       
#   |_|__\_\______/_/_ __\_\_____/__ _   |_|__|_|_ |_|______|_ _ _ 
#  / ____/ __ \|  \/  |  \/  |  ____| \ | |__   __/ ____| | | | | |
# | |   | |  | | \  / | \  / | |__  |  \| |  | | | (___   | | | | |
# | |   | |  | | |\/| | |\/| |  __| | . ` |  | |  \___ \  | | | | |
# | |___| |__| | |  | | |  | | |____| |\  |  | |  ____) | |_|_|_|_|
#  \_____\____/|_|  |_|_|  |_|______|_| \_|  |_| |_____/  (_|_|_|_)
#                                                                  
#
#
# This is a CMake makefile.  CMake is a tool that helps you build C++ programs.
# You can download CMake from http://www.cmake.org.  This CMakeLists.txt file
# you are reading builds dlib's example programs. 
#


cmake_minimum_required(VERSION 3.10.0)
# Every project needs a name.  We call this the "examples" project.
project(examples)


# Tell cmake we will need dlib.  This command will pull in dlib and compile it
# into your project.  Note that you don't need to compile or install dlib.  All
# cmake needs is the dlib source code folder and it will take care of everything.
add_subdirectory(../dlib dlib_build)

# If you have cmake 3.14 or newer you can even use FetchContent instead of
# add_subdirectory() to pull in dlib as a dependency.  So instead of using the
# above add_subdirectory() command, you could use the following three commands
# to make dlib available:
#  include(FetchContent)
#  FetchContent_Declare(dlib
#      GIT_REPOSITORY https://github.com/davisking/dlib.git
#      GIT_TAG        v19.24
#  )
#  FetchContent_MakeAvailable(dlib)

# To compile this program all you need to do is ask cmake.  You would type
# these commands from within the directory containing this CMakeLists.txt
# file:
#   mkdir build
#   cd build
#   cmake ..
#   cmake --build . --config Release
#
# The cmake .. command looks in the parent folder for a file named
# CMakeLists.txt, reads it, and sets up everything needed to build program.
# Also, note that CMake can generate Visual Studio or XCode project files.  So
# if instead you had written:
#   cd build
#   cmake .. -G Xcode
#
# You would be able to open the resulting Xcode project and compile and edit
# the example programs within the Xcode IDE.  CMake can generate a lot of
# different types of IDE projects.  Run the cmake -h command to see a list of
# arguments to -G to see what kinds of projects cmake can generate for you.  It
# probably includes your favorite IDE in the list.




#################################################################################
#################################################################################
#  A CMakeLists.txt file can compile more than just one program.  So below we
#  tell it to compile the other dlib example programs using pretty much the
#  same CMake commands we used above.
#################################################################################
#################################################################################

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Since there are a lot of examples I'm going to use a macro to simplify this
# CMakeLists.txt file.  However, usually you will create only one executable in
# your cmake projects and use the syntax shown above.
macro(add_example name)
   add_executable(${name} ${name}.cpp)
   target_link_libraries(${name} dlib::dlib )

   if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.13.0") 
   # And strip symbols to make your binary smaller if you like.  Certainly not
   # required though.
       target_link_options(${name} PRIVATE $<$<CONFIG:RELEASE>:-s>)
   endif()
endmacro()

# if an example requires GUI, call this macro to check DLIB_NO_GUI_SUPPORT to include or exclude
macro(add_gui_example name)
   if (DLIB_NO_GUI_SUPPORT)
      message("No GUI support, so we won't build the ${name} example.")
   else()
      add_example(${name})
   endif()
endmacro()

if (DLIB_NO_GUI_SUPPORT)
   message("No GUI support, so we won't build the webcam_face_pose_ex example.")
else()
   find_package(OpenCV QUIET)
   if (OpenCV_FOUND)
      include_directories(${OpenCV_INCLUDE_DIRS})

      add_executable(basic_capture basic_capture.cpp)
      target_link_libraries(basic_capture dlib::dlib ${OpenCV_LIBS} )
      add_executable(capture_detect capture_detect.cpp)
      target_link_libraries(capture_detect dlib::dlib ${OpenCV_LIBS} )
   else()
      message("OpenCV not found, so we won't build the webcam_face_pose_ex example.")
   endif()
endif()

find_package(nadjieb_mjpeg_streamer REQUIRED)
add_executable(web_capture web_capture1.cpp)                  
target_link_libraries(web_capture dlib::dlib ${OpenCV_LIBS} nadjieb_mjpeg_streamer::nadjieb_mjpeg_streamer) 
