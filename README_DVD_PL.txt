Aby uruchomić program, należy najpierw zainstalować biblioteki pomocnicze:

numpy w wersji co najmniej 1.11.0
theano w wersji co najmniej 0.8.0
matplotlib
nose
python-opencv (Napotkaliśmy problemy z instalacją opencv przez program pip.
               Można zainstalować z pomocą menadżera pakietów systemu Ubuntu.)

Następnie należy zainstalować bibliotekę Athenet komendą
python setup.py develop
(zgodnie z instrukcjami w README.md w katalogu Athenet).


Po zainstalowaniu oprogramowanie jest gotowe do użytku. Umożliwia tworzenie
własnych sieci neuronowych, trenowanie ich i uruchamianie na nich algorytmów zerujących wagi.

Aby uruchamiać algorytmy na przygotowanych sieciach neuronowych: LeNet, AlexNet,
GoogLeNet należy pobrać wagi do tych sieci. Należy również pobrać zbiory testowe,
na których testowana jest skuteczność działania zmodyfikowanych sieci.

Wagi i zbiór testowy MNIST do sieci LeNet są pobierane automatycznie. Pierwsze użycie
sieci LeNet wymaga dostępu do internetu.

Teraz jest już możliwe uruchamianie wszystkich algorytmów zerujących wagi na sieci
LeNet. Najprostszym sposobem uruchomienia algorytmów jest użycie programu znajdującego
się w Athenet/demo/indicators.py. Instrukcję obsługi programu można odczytać
uruchamiając ten skrypt w następujący sposób:

python indicators.py -h

Wagi do sieci AlexNet i GoogLeNet oraz potrzebny dla nich zbiór testowy ImageNet 2012 są
znacznie większymi plikami i wymagają ręcznego pobrania. Uruchamianie algorytmów dla tych
sieci wymaga dużej ilości ramu (32GB-64GB) i dużej mocy obliczeniowej. Działanie na sieci
LeNet nie jest aż tak wymagające. Opis kroków w celu uruchomienia tych sieci znajduje się
w plikach Athenet/alexnet_readme.txt oraz Athenet/googlenet_readme.txt. Potrzebne pliki
będą hostowane co najmniej do października 2016.



Opis plików demonstracyjnych w Athenet/demo:

algorithms.py pozwala uruchomić większość napisanych algorytmów, bez możliwości
łączenia ich ze sobą. Wspiera algorytmy Sender i Sender2, które nie są wspierane
przez inne skrypty.

indicators.py pozwala uruchomić algorytmy oprócz algorytmów Sender i Sender2.
Pozwala również łączyć algorytmy i uruchamiać je na konkretnych typach wag.
Dla zadanej konfiguracji optymalizuje obliczenia dla wielu ułamków wag
do wyzerowania, podawanych jako parametr do każdego z algorytmów.

derest_demo.py pozwala uruchomić algorytm Derest dla małej sieci dla ustalonej
konfiguracji.

mnist_learning_lenet.py pozwala na wytrenowanie sieci LeNet na podstawie zbioru
testowego.

run_derest.py uruchamia Derest dla różnych konfiguracji. Nie da się go uruchomić
na maszynie z małą ilością RAMu. Pewne uwagi na ten temat są wspomniane
w run_derest_readme.txt
