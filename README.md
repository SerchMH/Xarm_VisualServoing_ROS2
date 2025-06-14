# Xarm_VisualServoing_ROS2

**Xarm_VisualServoing_ROS2** es un paquete ROS 2 para implementar visual servoing en robots xArm usando retroalimentación basada en cámaras. Permite control cartesiano en tiempo real con integración de tf2 y compatibilidad con MoveIt 2.

---

## 🚀 Características

- Servoing visual en tiempo real utilizando datos de cámaras RGB-D o estéreo  
- Control cartesiano del efector final del robot  
- Soporte para movimientos en el plano X-Y  
- Transformaciones de frames con tf2 (por ejemplo, `camera_link` → `base_link`)  
- Integración con MoveIt 2 para planificación y ejecución de movimientos  
- Diseño modular compatible con diferentes cámaras y configuraciones sensoriales  

---

## 🧩 Requisitos del sistema

- ROS 2 Humble (o superior)  
- Repositorios oficiales de ROS 2 para xArm y Azure Kinect:
  - [xArm ROS2](https://github.com/xArm-Developer/xarm_ros2)
  - [Azure Kinect ROS2](https://github.com/microsoft/azure_kinect_ros)
- Cámara RGB-D o estéreo compatible (RealSense, ZED, Kinect, etc.)  
- Python 3.8 o superior  
- [MoveIt 2](https://moveit.ros.org/) para planificación de movimientos  

---

## 📁 Estructura del repositorio

Xarm_VisualServoing_ROS2/
├── src/ # Código fuente principal
├── build/ # Archivos temporales de compilación
├── install/ # Archivos instalados
├── log/ # Archivos de registro
├── .vscode/ # Configuración de VSCode (opcional)
└── README.md # Este archivo


---

## 🧠 Descripción general

Este paquete utiliza retroalimentación visual (por ejemplo, detección de centroides o marcadores ArUco) para estimar la pose del objetivo en coordenadas de la cámara, transformar esa pose al marco base del robot y generar comandos de movimiento que guían el efector final hacia el objetivo. El servoing puede realizarse con objetivos de posición o mediante control de velocidad.

Para la adquisición de datos visuales, se emplean los repositorios oficiales de Azure Kinect ROS2 y xArm ROS2, facilitando la integración fluida con cámaras Azure Kinect y el robot xArm respectivamente.

---

## 📚 Documentación y recursos

- [Repositorio oficial xArm ROS 2](https://github.com/xArm-Developer/xarm_ros2)  
- [Repositorio oficial Azure Kinect ROS2](https://github.com/microsoft/azure_kinect_ros)  
- [Documentación MoveIt 2](https://moveit.picknik.ai/)  
- [Tutoriales ROS 2](https://docs.ros.org/)  

---

## 👨‍💻 Autores

- **Sergio Muñoz Hiromoto** – Ingeniería en Robótica y Sistemas Digitales, Tec de Monterrey  
- **Alejandro Rodriguez del Bosque** – Ingeniería en Robótica y Sistemas Digitales, Tec de Monterrey  
- **Alfonso Solis Diaz** – Ingeniería en Robótica y Sistemas Digitales, Tec de Monterrey  

---

## 📜 Licencia

Este proyecto está bajo la **Licencia MIT**. Consulta el archivo [LICENSE](LICENSE) para más detalles.
