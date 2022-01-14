# (WIP) ESPECTRO DE MODULAÇÃO APLICADO AO PROCESSAMENTO DE FALA 
Trabalho de conclusão de curso para obtenção do grau de Engenheiro Eletrônico e da Computação pela Universidade Federal do Rio de Janeiro.

## Tema
O tema do trabalho é o espectro de modulação, que possibilita a
 análise de oscilações de segunda ordem em sinais, de modo a abordar técnicas de
  *speech enhancement* que: 1. estendem métodos de redução de ruído no
 domínio da transformada de Fourier de tempo curto (STFT, do inglês
 *short-time Fourier transform*) ao domínio do espectro de modulação, e
 2. a técnica de filtragem de modulação, que visa à
 decomposição e filtragem de frequências moduladoras com vistas a maior
 inteligibilidade da fala.
 

## Justificativa

Aplicações que dependam de captação de voz, como chamadas de voz sobre IP (VoIP,
 do inglês *voice over IP* e reconhecimento de fala, podem ter sua
 experiência de uso e assertividade melhoradas quando a fala captada é mais
 inteligível, o que pode ser potencializado com técnicas de \*speech
 enhancement* aplicadas sobre o áudio analisado.

No conjunto das técnicas de *speech enhancement*, as que utilizam a
transformada discreta de Fourier (DFT, do inglês *discrete Fourier
transform*) na forma de uma transformada rápida de fourier (FFT, do inglês
*fast Fourier transform*) apresentam baixa complexidade computacional,
além de permitirem a modificação dos coeficientes do sinal no domínio da
frequência e sua ressíntese no domínio do tempo. Entretanto, o uso da DFT é mais
adequado para sinais estacionários. Sinais quase estacionários, isto é, sinais
cuja estatística é aproximadamente constante em curtos períodos, tais como a
fala —-- em janelas da ordem de milissegundos —--, podem ser bem representados
pela transformada de Fourier de curta duração (STFT, do inglês
*short-time Fourier transform*). Sobre esta, é possível utilizar
algoritmos de redução de ruído, como a subtração espectral e o filtro de
Wiener. Uma das técnicas evoluídas a partir da STFT é o
espectrograma de modulação —-- objeto da presente pesquisa —--, que representa
oscilações de segunda ordem no sinal de áudio analisado, no chamado domínio da
modulação.

O sinal de fala pode ser representado como a sobreposição de portadoras geradas
pelas cordas vocais, cujas amplitudes e frequências variam lentamente em
consequência das mudanças provocadas pelo trato vocal e de seus articuladores,
durante a fonação. A bibliografia demonstra que a componente AM contribui para a
inteligibilidade do sinal de fala, uma vez que a envoltória quantiza a estrutura
temporal de fonemas, sílabas e frases, atribuindo ritmicidade a essas
unidades. Quanto à percepção auditiva, atribui-se
à cóclea a capacidade de filtrar o som (de banda larga), em diversas sub-bandas
de banda estreita, de forma que modulações em amplitude sobre cada sub-banda
sejam passadas adiante no sistema auditivo. Portanto, o espectrograma de
modulação demonstra-se adequado para aplicações de *speech enhancement*,
pois permite a representação da modulação em um domínio que considere diferentes
frequências de portadora --— de forma análoga ao ``banco de filtros'' da cóclea
--— e representa, tal como a STFT, a quase estacionariedade em curtos períodos
de tempo característica em sinais de fala. Avaliações do desempenho de técnicas
de *speech enhancement* que atuam no domínio do espectrograma de
modulação reforçam essa justificativa, indicando bom desempenho em índices de
inteligibilidade.

Uma segunda técnica disponível é a filtragem de modulação. Essa
técnica se baseia nas evidências de que, ao aumentar gradualmente a frequência
de corte para um filtro passa-baixas aplicado sobre oscilações da envoltória de
um sinal, as componentes acima de 16Hz apresentam apenas um incremento marginal
na inteligibilidade. Dessa forma, as componentes de
modulação abaixo do limiar de 16Hz são suficientes para uma boa compreensão da
fala. A partir dessa característica da percepção, o filtro de modulação é capaz
de limitar ruídos sobre a envoltória que incorrem fora da região filtrada, dessa
forma aumentando a inteligibilidade.


Por fim, a bibliografia que aborda essas técnicas, apesar de circular há cerca
de 20 anos em pesquisa, possui poucos documentos que agregam, condensam e
abordam sua evolução e suas formas mais sofisticadas, que carecem de uma
apresentação da teoria em que se baseiam ao alcance de um leitor
interessado.

## Objetivo

O objetivo geral do projeto é apresentar de forma acessível o espectrograma de
modulação e a filtragem de modulação, juntamente com uma aplicação prática sua
em processamento de áudio. 

Os objetivos específicos são: (1) Apresentar a filtragem de modulação. (2)
Apresentar a teoria do espectrograma de modulação. (2) Apresentar técnicas de
*speech enhancement* (ex.: filtro de Wiener) no domínio da STFT. (3)
Comparar seu desempenho quando aplicadas no espectrograma de amplitude e no
espectrograma de modulação. (4) Apresentar uma aplicação, já presente na
bibliografia, que aborde as duas técnicas comentadas. Dessa forma, o trabalho
servirá como material de consulta para pesquisadores que se interessem pelo
tema.
