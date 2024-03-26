<h1>Documento resumen del proyecto para determinar el tiempo que hace en imágenes</h1>

<ol>
<h2><li>Resumen del proyecto</li></h2>
<p>Este proyecto consistirá en la creación de varias redes neuronales con diferentes tecnologías para clasificar diferentes imágenes en función del tiempo que se puede apreciar en ellas. Una red neuronal convolucional es un tipo de red neuronal artificial especialmente diseñada para procesar datos que tienen una estructura de cuadrícula, como imágenes o datos de series temporales. En este proyecto se van a crear varias y se van a tratar de optimizar sus parámetros. </p>








<h2><li>Tecnologías utilizadas</li></h2>
<p>Se van a utilizar las siguientes tecnologías relacionadas con Python:</p>
<ul>
<li>Tensorflow</li>
<li>Keras</li>
<li>PyTorch</li>
<li>FastAI</li>
</ul>

<h2><li>Descripción del conjunto de datos</li></h2>
El conjunto de datos se ha obtenido de https://www.kaggle.com/datasets/jehanbhathena/weather-dataset/data donde se pueden encontrar varias imágenes separadas en diferentes carpetas agrupándolas por el tipo de tiempo que se puede apreciar en ellas. Estas imágenes se pueden clasificar en rocío, niebla toxica, cencellada, hielo, granizo, relámpagos, lluvia, arcoíris, escarcha, tormenta de arena y nieve.






<h3>Separación conjunto de datos y de test</h3>
El conjunto de datos original con todas las imágenes es muy grande, con alrededor de 6900 imágenes divididas en las 11 clases comentadas previamente. Para este proyecto, se han seleccionado para cada clase 23 imágenes en cada clase para el conjunto de entrenamiento, y 12 imágenes de cada clase para el conjunto de validación y 12 imágenes de cada clase para el de test.


<h2><li>Optimización de los parámetros</li></h2>
Para la optimización de parámetros a continuación se definen los parámetros optimizados en cada caso. El proceso de optimización de parámetros se realiza mediante varios bucles anidados que recorren cada combinación diferente de parámetros y crea el modelo aplicándolos. Una vez hecho esto, obtiene la precisión del modelo sobre el conjunto de test y se guarda en un dataframe la combinación de parámetros establecida y la precisión obtenida. Como son muchas para cada modelo el número de combinaciones posibles de parámetros se ha optado por ir pivotando con el parámetro del número de épocas, de manera que se evalúen todas las combinaciones de parámetros con el mismo número de épocas y se genere el dataframe con los resultados. Después se evalúan las combinaciones de parámetros con otro número de épocas y se genera el dataframe correspondiente. De esta manera estos dataframe se pueden ir almacenando en un excel y la ejecución del programa no es excesivamente larga.

<h3>Optimización parámetros </h3>
A continuación, se detallan los parámetros a optimizar en la red neuronal creada con FastAI:
<ul>
<li>Metrica: accuracy, error_rate, Precision(average='macro'), Recall(average='macro')</li>
<li>Épocas: 4, 6, 8</li>
<li>Batch_size: 4, 8, 12</li>
</ul>

La tabla con todos los valores sobre la precisión para las diferentes combinaciones de parámetros en este modelo se puede encontrar en el Excel adjunto “erroresFastAI.xlsx”. Por lo tanto, los mejores valores en los parámetros para este modelo son los siguientes:
<ul>
<li>Metrica: accuracy</li>
<li>Épocas: 8</li>
<li>Batch_size: 8</li>
</ul>



A continuación, se detallan los parámetros a optimizar en la red neuronal creada con Pytorch:
<ul>
<li>Optimizador: SGD, Adam, Adagrad</li>
<li>Épocas: 4, 6, 8</li>
<li>Batch_size: 4, 8, 12</li>
<li>Capas: 3, 4, 5</li>
</ul>

La tabla con todos los valores sobre la precisión para las diferentes combinaciones de parámetros en este modelo se puede encontrar en el Excel adjunto “erroresPytorch.xlsx”. Por lo tanto, los mejores valores en los parámetros para este modelo son los siguientes:
<ul>
<li>Optimizador: Adagrad</li>
<li>Épocas: 6</li>
<li>Batch_size: 4</li>
<li>Capas: 3</li>
</ul>


A continuación, se detallan los parámetros a optimizar en la red neuronal creada con Tensorflow y keras:
<ul>
<li>Optimizador: SGD, Adam, Adagrad</li>
<li>Épocas: 4, 6, 8</li>
<li>Batch_size: 4, 8, 12</li>
<li>Capas: 2, 3, 4</li>
</ul>

La tabla con todos los valores sobre la precisión para las diferentes combinaciones de parámetros en este modelo se puede encontrar en el Excel adjunto “erroresTensorKeras.xlsx”. Por lo tanto, los mejores valores en los parámetros para este modelo son los siguientes:
<ul>
<li>Optimizador: Adam</li>
<li>Épocas: 8</li>
<li>Batch_size: 12</li>
<li>Capas: 2</li>
</ul>



Finalmente, se ha podido apreciar que el mejor resultado se ha obtenido con la red neuronal creada con FastAI con una precisión de 0.788.


<h2><li>Resultado final: vídeo youtube y repositorio</li></h2>
Repositorio Github:


<h2><li>Conclusiones</li></h2>
He podido aprender a configurar los diferentes parámetros de una red neuronal y como crearlas para un problema de clasificación de imágenes con diferentes tecnologías.

</ol>
