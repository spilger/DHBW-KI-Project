# DHBW-KI-Project

Dog Dataset: http://vision.stanford.edu/aditya86/ImageNetDogs/


 Motivation  
 Methodik  
 Datenquellen  
 Algorithmus 
 Ergebnisse 
 Diskussion  
 
 Pro Thema ca. 1 Seite 


Prüfungsleistung Interview (V8)Abgabe Bericht (33%)  

Motivation  
Methodik  
Datenquellen  
Algorithmus 
Ergebnisse 
Diskussion  

Pro Thema ca. 1 Seite 30s Video (33%) (Für Investoren ohne technischen Hintergrund) Begeistere Investoren für deine Idee Nutzen  Umsetzung


InterviewCa 30 Minuten  Wie habt ihr das umgesetzt  Besprechung des Berichts  TG stellt Fragen zu DetailsAlgorithmusDatenverarbeitung Etc.Interview ca. 30 Minuten


# Ausarbeitung
Motivation (Anna-Lena)
*	Versicherung
*	Hund Tierheim
*	Interessen
*	Ziel: App 
* Zielgruppe: Leute mit Handy
* Anwendung: Foto machen oder Hochladen (Versicherung)
* Zielstellung: die wichtigsten 90%? Hunderassen erkennen
* Fragestellung: Können Mischlinge erkannt werden?

Datenquellen (Jessica)
*	Woher sind die Daten
*	Was ist Stanford
*	Wie sehen die Daten
*	Aufbau der Daten
*	Beispielhafte Bilder zeigen

Methodik (Stichpunktartige Vorgehensweise von Micha)
*	Vorgehensweise
*	* Recherche zur Strukturierung neuronaler Netze zur Bild-Klassifizierung
*	* Test zum finden einer Struktur zur Klassifikation von 5 Bildklassen (Struktur des ersten Netzes)
*	* Erfolg - klappt mit 97% Validation Accuracy
*	* Ok aber schafft das Netz auch mehr Klassen
*	* Bei 120 Klassen starkes Overfitting
*	* Lösungsansatz: Bilder in jedem Trainingsschritt zufällig verändern (zoomen, drehen, spiegeln)
*	* Erfolg: Modell ist nicht mehr overfitted - Problem die Accuracy ist nur noch niedrig bei circa 12% in der Validation
*	* Lösungsansatz: Komplexität mit Dense Layern und Conv2D-Layern ehöhen: Problem Trainingsaufwand mit verfügbaren Ressourcen nicht mehr möglich
*   * Lösungsansatz: Nur die Komplexität eines Dense Layers nach den Conv2D Layers erhöhen. - Besser, aber immer noch großer Aufwand
*	* Suche nach Lösungsmöglichkeiten: Idee vortrainierte Image-Classifier-Netze in Netzstruktur integrieren
*	* Zwei Funktionierende Ansätze gefunden: 
*	* * 1. DenseNet32_121 in Netzstruktur integrieren circa 20 Stunden Trainingsaufwand für circa 85% Validation accuracy
*	* * 2. EfficientNet über tflite_modelmaker - neues Tensorflow Modul bietet nativ integririert Image_classifier Netze zum weitertrainieren an Test mit EfficientNet121 - Erfolg circa 80% validation accuracy und geringer Trainingsaufwand circa 1 Stunde mit verfügbaren Ressourcen
*	* Integration in App zur lokalen Ausführung
*	Anfang: Overfitting (zu sehr angepasst an die Testdaten)
*	Lösung: zufälliges Drehen, Zoomen, Flippen der Bilder
*	Validation accuracy: wurde besser, aber noch nicht optimal
*	Vorgefertigtes Modell verwenden + Flippen der Bilder: Validierung sogar besser

Algorithmus (Jessica) 
*	Tensorflow light
*	Neuronale Netze
*	https://arxiv.org/pdf/1905.11946.pd 
*	https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-tensorflow/ 
*	Code einblenden und erklären

Ergebnisse (Anna-Lena)
* Epochen: erhöhen ==> ab gewisser Anzahl stagnieren wird nur der Test besser (Überanpassung), Stopp: Maximum in der Validationaccuray
*	Accuracy beschreiben: Training + Validierung
*	Grafik: Verlauf Genauigkeit (Epochen x-Achse, Y-Achse: Genauigkeit)
*	Verweis Methodik: Kein Overfitting
* Loss beschreiben: Grafik
* App reinbringen: Beschreibung, Screenshot

Diskussion
*	Objektiv betrachtete Ergebnisse 
* Bezüge herstellen
*	Interpretieren
*	Mutmaßungen
*	Was würde passieren, wenn das und das passieren würde
*	Zielstellung
*	Was haben uns die Ergebnisse gebracht
*	Fazit und Ausblick

* Schwachstelle: Mischlinge (z. B. Labradudel) können nicht immer zuverlässig erkannt => Lösung: Mischlingsbilder
* Schwachstelle: 350? anerkannte Hunderasse, wir betrachten nur 120? => Lösung: die Hunderassen miteinbezieht => Aber: Wir decken 90%? der wichtigsten Hund ab. mehr Klassen = anderes Modell verwenden und Overfitting vermeiden
* Zielstellung: die wichtigsten 90%? Hunderassen erkennen + Mischlinge teilweise erkennen unter bestimmten Bedingungen
* EffificientNet: Leistungsfähige Ressourcen weiternutzen => Nicht von 0 angefangen (wäre dabei das gleiche Resultat möglich gewesen unter Betrachtung Ressourcen (Zeit, Aufwand, Effizienz, Genauigkeit) 


