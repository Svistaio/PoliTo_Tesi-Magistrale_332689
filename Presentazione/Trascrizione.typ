
//#region Preambolo
    #import "Preambolo.typ":*

    #show:articleTemplate
    #show:mathEnvTemplate
//#endregion

//

//#region Note
  #block(
    width:100%,
    height:1cm,
    inset:5pt,
    stroke:draftStroke(),
    below:2em,
  )[
    #set text(size:article-sizes.title)
    #align(center+horizon)[
      // _Appunti della presentazione_
      _Trascrizione della presentazione_
    ]
  ]

    #set enum(numbering:"[1]",indent:-2.4em)
    #enum[
      Benvenuti alla presentazione della tesi magistrale sulla «Modellizzazione della distribuzione della popolazione tra città su reti spaziali mediante la teoria cinetica dei sistemi multiagente».
    ][
      Come suggerisce il titolo si parla delle taglie delle città, e il primo a popolarizzare un tale studio fu Zipf, seppure in un contesto linguistico, formulando la seguente legge empirica in cui la frequenza delle parole $f$ è inversamente proporzionale al loro rango $r$, ossia alla loro posizione nella classifica; essa vale, però, anche in un contesto urbano sostituendo $f$ con $s$, ma solo per $s>>1$, ovvero solo sulla coda della distribuzione, ed è pure generalizzabile mediante una distribuzione di Pareto dove la FRC, relativa alla distribuzione della popolazione $f_S$, è l'analogo del rango mentre $beta$ è l'indice di Pareto.
      //; [*Lavanga*]{per esempio se si considera la decima parola ($r=10$) che compare 2000 ($f=2000$) allora esiste $c in RR$ tale che $2000=c\/10$}
    ][
      Piú recentemente si è scoperto che l'intera distribuzione è ben fittata da una bilognormale, ossia una combinazione convessa di due lognormali aventi la seguente densità; questa assomiglia a una normale e ...
    ][
      ... in effetti plottando alcune lognormali, tutte collo stesso valore atteso, si può notare che, passando dallo spazio lineare a quello logaritmico $y=log(x)$, esse diventano a tutti gli effetti delle normali, da cui il nome.// Si noti anche come in spazio logaritmico i parametri $mu$ e $sigma$ coincidano proprio col valore atteso e la deviazione della normale, mentre in spazio lineare tutte queste hanno lo stesso valore atteso.
      // [*Lavagna*]{$y=log(x)$}
    ][
      Dunque l'obbiettivo è il seguente: da un punto di vista metodologico si vogliono simulare attraverso la TCSMA (M1) le interazioni tra città (M2), in particolare vedendo queste come agenti piuttosto che luoghi, com'è piú comune nella letteratura, su un grafo spaziale (M3); invece per i risultati si vogliono riprodurre le due caratteristiche tipiche delle distribuzioni reali: corpo bilognormale e coda di Pareto.
    ][
      Prima di analizzarli è necessario un po' di contesto teorico iniziando con alcuni cenni di teoria delle reti. Si parte dalla definizione di grafo come coppia d'insiemi: uno degl'indici e l'altro dei lati; da quest'ultimo si può definire la matrice d'adiacenza pesata che associa a ogni lato un certo peso $q_(i,j)$ la cui somma totale su una colonna o una riga fornisce rispettivamente la forza entrante e uscente. Nel caso il peso sia unitario si è soliti indicare M con A e chiamare le forze i gradi, ma sono concetti del tutto affini.
    ][
      D'altra parte le ipotesi sul grafo sono due. Innanzitutto è diretto perché se, per esempio, la prima città interagisce colla terza non è detto che questa interagisca colla prima, ossia è necessario considerare il senso di direzione dell'interazione; poi è simmetrico poiché se un'interazione è possibile lo sarà anche l'opposta quindi ogni lato implica l'inverso, da cui la simmetria; ciò conferisce al grafo diretto una struttura indiretta visto che condivide la stessa matrice d'adiacenza. infine è statico sia per mancanza di dati storici precisi rispetto cui confrontare i risultati, ma, soprattutto, per semplicità analitica.
    ][
      Per quanto riguarda la teoria cinetica, il modello si può riassumere essenzialmente con tre elementi: il primo sono le regole d'interazione che descrivono il trasferimento della frazione $E$ dalla città interagente $S$ a quella ricevente $S_*$, dove $E$ rappresenta la regola d'emigrazione da definire euristicamente per completare le $BracketRef("RI","S")$. Tuttavia non è detto che l'interazione avvenga, comportamento che si può modellizzare mediante una distribuzione di Bernoulli, nella quale è infatti presente l'elemento corrispettivo ai due nodi della matrice d'adiacenza: se non sono connessi non può avvenire l'interazione; unendo questi due elementi si ricava l'algoritmo di azione-reazione, chiamato cosí perché se due agenti interagiscono allora entrambi alterano il proprio stato.
    ][
      Dall'$#BracketRef("AR","S")$, e avvalendosi di un osservabile $Phi$, si perviene a quest'equazione di tipo Boltzmann che descrive l'evoluzione nel tempo della distribuzione statistica $f$ del microstato $(I,S_t)$; piú nel dettaglio il primo membro è la forma debole di un operatore di trasporto lineare omogeneo nello spazio mentre il secondo membro è la forma debole di un operatore collisionale.
    ][
      Il modello appena definito si può anche approssimare mediante $bold(B)$ definita come il prodotto diadico tra il vettore dei gradi riscalato dalla norma unitaria della matrice d'adiacenza; l'equazione di tipo Boltzmann cosí ottenuta è analoga alla precedente ma con $bold(B)$ al posto di $bold(A)$. Si nota che $bold(B)$ approssima $bold(A)$ solo in quanto condivide i suoi gradi uscenti ed entranti per ogni nodo, ossia $bold(B 1)=bold(A 1)$ e $bold(B^top 1)=bold(A^top 1)$, ma per il resto ne differisce completamente. Eppure ci si potrebbe domandare perché non si possa scegliere un'altra matrice $bold(C)$ che approssima $bold(A)$ secondo altri criteri; certamente si potrebbe fare, ma ciò che distingue $bold(B)$ è che la sua equazione di tipo Boltzmann può essere riformulata al limite in un'altra in cui lo stato dell'indice $I$ è diventato il grado $K$.
    ][
      Per comprendere a pieno il significato alquanto teorico di tale trasformazione è meglio rappresentarla graficamente: di fatto si parte da una topologia esatta, descritta da $bold(A)$, quindi mediante $bold(B)$ si arriva a una approssimata (per semplicità visiva il grafo è indiretto anziché diretto) avente però lo stesso microstato, infine si giunge a una nella quale non è piú possibile distinguere i singoli nodi perché il microstato perde l'indice in luogo del grado: si definisce dunque una classe di grafi tutti con uguale distribuzione dei gradi, rilassando in tal modo la topologia. Ciò permette di avvicinarsi di piú al contesto particellare originale da cui è nata l'equazione di tipo Boltzmann.
    ][
      Con tutto ciò si può parlare dei risultati iniziando illustrando il metodo di Monte Carlo coi quali sono stati ottenuti; questa è una scelta tipica nella teoria cinetica dei sistemi multiagente: infatti, rispetto ad altri metodi, è quello piú fisico siccome permette di recuperare la fisica particellare soggiacente all'equazione di Boltzmann. A grandi linee si parte da un vettore iniziale $pzclS^0$, uniforme rispetto alla popolazione totale $P$, quindi per ogni passo temporale si permutano indipendentemente i primi $N$ indici, poi si accoppiano i primi $N\/2$ agenti coi restanti, quindi se l'interazione avviene ($Theta=1$) si modificano le taglie seguendo esattamente l'$[text("AR")]_text("S")$, altrimenti si lasciano invariate. Infine l'istogramma del vettore in ogn'istante approssima la vera distribuzione. È anche ovvio che, data la natura stocastica dell'algoritmo, sia necessario considerare piú simulazioni per ottenere un risultato statisticamente significativo.
    ][
      I primi risultati sono stati ottenuti tramite questa regola d'emigrazione ispirata da una funzione di Hill di ordine $alpha$, comunemente applicata nel campo della Biomedicina: in poche parole descrive che il tasso d'emigrazione è maggiore verso città piú popolate. Empiricamente si è verificato che esistono solo due configurazioni: una, per $alpha$ bassi, è stabile ma banale, poiché la distribuzione rimane intorno a quella iniziale uniforme, mentre l'altra, per $alpha$ sufficientemente alti, è instabile ed è quella qui raffigurata, da cui si evince che esistono delle classi di gradi che si spopolano.
    ][
      Le distribuzioni confermano quanto detto mostrando che vi sono delle realizzazioni nelle quali alcune città raggiungono una popolazione di $10^(-50)$, mentre le dispersioni identificano la classe spopolata coincidente coi centri mediamente connessi. L'unico lato positivo è la buona corrispondeza tra la simulazione esatta e approssimata ma nel complesso si può affermare che il fenomeno della migrazione molto probabilmente non dipende _solo_ dal fattore della popolazione relativa.
    ][
      Si può allora pensare di modificare la precedente regola aggiungendo anche il grado relativo: cosí ragionando il tasso d'emigrazione è maggiore verso città sia piú popolate che piú connesse. Da questa configurazione i risultati sono già migliori perché convergono non mostrando segni di spopolamento.
    ][
      Invece le distribuzioni approssimano bene quella reale ricavata dai dati ISTAT, mentre le dispersioni riescono a cogliere la correlazione positiva sebbene sia molto piú stretta rispetto a quella reale, assomigliando di piú a una relazione biunivoca tra popolazione e grado. Confrondando inoltre le due simulazioni, si può vedere come queste siano quas'identiche, confermando la bontà dell'approssimazione.
    ][
      Si può allora analizzare l'indice di Pareto: quello approssimato ed esatto sono tra di loro pressoché uguali, ed entrambi sono coerenti con quello reale differenziandosi solo al second'ordine. D'altra canto, confrontando Pareto e bilognormale, si può notare in tutt'i casi come la seconda segua meglio la coda rispetto alla prima seppure nella prima metà coincidano. Dunque da questi risultati si può dire che il fenomeno della migrazione molto probabilmente dipende da _almeno_ questi due fattori relativi.
    ][
      È rilevante sottolineare che la convergenza della #BracketRef("RE","SK") non è scontata. Per mostrarlo si considerino i due nodi in figura _in condizioni stazionarie_ ove quello a sinistra è relativamente maggiore rispetto a quello a destra; di conseguenza è chiaro che s'instauri un flusso accentrante da $i_*$ verso $i$, ma, essendo la configurazione stabile, deve anche sussistere un flusso inverso che bilancia il precedente; eppure, siccome la funzione di Hill è crescente, ci si rende conto che il suo valore è minore colla nuova regola d'emigrazione, la quale quindi manifesta un retroflusso piú piccolo. Da questo ci si aspetterebbe che il modello sia piú instabile; cionnonostante converge rendendo tale comportamento una stimolante domanda teorica aperta.
    ][
      Tuttavia il grado è un informazione puramente binaria: è la somma della presenza o assenza di tutt'i lati di un nodo, e non riesce cosí a cogliere effettivamente l'uso di una rete di trasporto; considerando allora la forza $w$ definita come il numero totale di pendolari _interurbani_ verso una città, si riescono a distinguire due centri con ugual grado ma diverso traffico interurbano. Da questa configurazione analogamente a prima i risultati convergono, ...
    ][
      ... le distribuzioni corrispondono quasi perfettamente a quella reale e lo stesso si può dire anche per le dispersioni che mostrano una correlazione sempre positiva ma meno stretta del caso precedente e piú in linea con quella reale. Queste ultime non sono però perfette perché, seppure lievemente, sovrastimano le taglie dei centri medio-minori e sottostimano quelle dei centri maggiori; ciò è molto probabilmente attribuibile alla mancanza del traffico _intraurbano_ nella definizione della forza $w$. Invece tra la simulazione esatta e approssimata, anche in questo caso non vi sono differenze significative riconfermando la bontà dell'approssimazione.
    ][
      Sulla Pareto e la bilognormale valgono gli stessi commenti di prima: tutt'i gl'indici, approssimato, esatto e reale, si differenziano tra di loro solo al second'ordine, da cui segue la loro somiglianza; anche in questo caso la bilognormale segue meglio la coda della distribuzione, seppure coincida colla Pareto nella prima metà. Pertanto è chiaro che il fenomeno della migrazione molto probabilmente è governato da questi due fattori relativi.
    ][
      È anche possibile interpretare i due parametri condivisi da tutte le regole finora viste tramite due studi parametrici che, ovviamente per brevità, non sono mostrati. Innanzitutto essi confermano che il valore di $lambda$, essendo un semplice riscalamento, modifica principalmente il tempo di convergenza, lasciando inalterata la distribuzione media; per tale ragione la sua variazione influisce maggiormente sull'intervallo $r>1$, ossia sull'attrazione esercitata dai centri relativamente maggiori. Viceversa modificare $alpha$ ha profonde ripercussioni su ogni aspetto dei risultati, specialmente sull'indice di Pareto; in effetti la sua variazione altera notevolmente la forma della funzione di Hill, soprattutto nell'intervallo $[0,1]$, vale a dire sulla repulsione sprigionata dai centri relativamente minori. È proprio per questa ragione che si può congetturare che $alpha$ sia inversamente legato al retroflusso stabilizzante: piú è grande, minore è il retroflusso e quindi piú instabile è il modello.
    ][
      Il lavoro qui presentato non è affatto esaustivo e conduce a molteplici sviluppi futuri. Se ne elencano per esempio quattro: 1) si possono definire regole d'emigrazione piú complesse, non necessariamente di Hill, usando pure altri coefficienti dalla teoria dei grafi; 2) si può tentare di ricavare l'equazione di Fokker-Planck per studiare sia la distribuzione stazionaria che il retroflusso stabilizzante; 3) si può considerare un grafo dinamico coevolvente colla distribuzione della popolazione; infine, 4) si può verificare questa teoria in altri contesti europei o internazionali.
    ][
      Grazie dell'attenzione.
    ]
    // [Eppure, a prescindere dai buoni risultati, ci si potrebbe chiedere se aggiungere il grado relativo sia una scelta modellisticamente sensata oppure una forzatura del modello a favorire città piú connesse.]
//#endregion
