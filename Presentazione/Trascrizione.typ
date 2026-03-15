
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

  // #boxed(
  //   blk-args:(
  //     below:2em,
  //   ),
  //   box-args:(
  //     inset:5pt
  //   ),
  // )[
  //   // #set text(size:article-sizes.section)
  //   #set par(leading: 4pt)
  //   Durante la presentazione molte informazioni non sono dette e vanno complementate o a voce o alla lavagna; per avere un riferimento è utile scrivere una lista che riassume quanto va detto.
  // ]

    #set enum(numbering:"[1]",indent:-2.4em)
    #enum[
      Benvenuti alla presentazione della tesi magistrale sulla «Modellizzazione della distribuzione della popolazione tra città su reti spaziali mediante la teoria cinetica dei sistemi multiagente».
    ][
      Come suggerito dal titolo si parla delle taglie delle città, e il primo a popolarizzare un tale studio fu Zipf, seppure in un contesto linguistico, attraverso la seguente legge empirica in cui la frequenza delle parole $f$ è inversamente proporzionale al loro rango $r$, ossia alla loro posizione nella classifica; essa vale, però, anche in un contesto urbano ma solo per $s>>1$, ovvero solo sulla coda della distribuzione, ed è pure generalizzabile mediante una distribuzione di Pareto dove la FRC, relativa alla distribuzione della popolazione $f_S$, è l'analogo del rango mentre $beta$ è l'indice di Pareto.
      //; [*Lavanga*]{per esempio se si considera la decima parola ($r=10$) che compare 2000 ($f=2000$) allora esiste $c in RR$ tale che $2000=c\/10$}
    ][
      Piú recentemente si è scoperto l'intera distribuzione è ben fittata da una bilognormale, ossia una combinazione convessa di due lognormali descritta dalla seguente densità; questa assomiglia a una normale e ...
    ][
      ... in effetti plottando in spazio lineare alcune lognormali, tutte condividenti lo stesso valore atteso, si può notare che passando in spazio logaritmico, ossia $y=log(x)$, esse diventano a tutti gli effetti delle normali, da cui il nome.// Si noti anche come in spazio logaritmico i parametri $mu$ e $sigma$ coincidano proprio col valore atteso e la deviazione della normale, mentre in spazio lineare tutte queste hanno lo stesso valore atteso.
      // [*Lavagna*]{$y=log(x)$}
    ][
      Dunque l'obbiettivo è duplice: da un punto di vista metodologico si vogliono simulare attraverso la TCSMA (M1) le interazioni tra città (M2), in particolare vedendo queste come agenti piuttosto che luoghi, com'è piú comune nella letteratura, su un grafo spaziale (M3); invece per i risultati si vogliono riprodurre le due caratteristiche tipiche delle distribuzioni reali: corpo bilognormale e coda di Pareto.
    ][
      Prima di analizzarli è necessario un po' di contesto teorico iniziando con alcuni cenni di teoria delle reti. Prima di tutto un grafo è una coppia d'insiemi: uno degl'indici e l'altro dei lati; da questi si può definire la matrice d'adiacenza pesata che associa a ogni lato un certo peso $q_(i,j)$ la cui somma totale su una colonna o una riga fornisce rispettivamente la forza uscente ed entrante. Nel caso il peso sia unitario si è soliti scivere A per M e chiamare i gradi le forze, ma sono concetti del tutto analoghi.
    ][
      D'altra parte le ipotesi sul grafo sono due. Innanzitutto è diretto perché se, per esempio, la prima città interagisce colla terza non è detto che questa interagisca colla prima, ossia è necessario considerare il senso di direzione dell'interazione; poi è simmetrico poiché se un'interazione è possibile lo sarà anche l'opposta quindi ogni lato implica l'inverso, da cui la simmetria; ciò conferisce al grafo diretto una struttura indiretta visto che condivide la stessa matrice d'adiacenza. infine è statico sia per mancanza di dati storici precisi rispetto cui confrontare i risultati, ma, soprattutto, per semplicità analitica.
    ][
      Per quanto riguarda la teoria cinetica, il modello si può riassumere essenzialmente con tre elementi: il primo sono le regole d'interazione che descrivono il trasferimento della frazione $E$ dalla città interagente $S$ a quella ricevente $S_*$, ed è per tale ragione che $E$ è denotata regola d'emigrazione; tuttavia non è detto che l'interazione avvenga, comportamento che si può modellizza mediante una distribuzione di Bernoulli, nella quale è infatti presente l'elemento corrispettivo ai due nodi della matrice d'adiacenza: se non sono connessi non può avvenire l'interazione; unendo questi due elementi si ricava l'algoritmo di azione-reazione, chiamato cosí perché se due agenti interagiscono allora entrambi alterano il proprio stato.
    ][
      Dall'$#BracketRef("AR","S")$, e avvalendosi di un osservabile $Phi$, si perviene a quest'equazione di tipo Boltzmann in cui $f$ è la distribuzione statistica del microstato $(I,S_t)$ a un tempo; piú nel dettaglio il primo membro è la forma debole di un operatore di trasporto lineare omogeneo nello spazio mentre il secondo membro è la forma debole di un operatore collisionale.
    ][
      Il modello appena definito si può anche approssimare mediante $bold(B)$ definita come il prodotto diadico tra il vettore dei gradi riscalato dalla norma unitaria della matrice d'adiacenza; l'equazione di tipo Boltzmann cosí ottenuta è analoga alla precedente ma con $bold(B)$ al posto di $bold(A)$. Si nota che $bold(B)$ approssima $bold(A)$ solo in quanto condivide i suoi gradi uscenti ed entranti per ogni nodo, ossia $bold(B 1)=bold(A 1)$ e $bold(B^top 1)=bold(A^top 1)$, ma per il resto ne differisce completamente. Ma allora perché non si potrebbe scegliere un'altra matrice $bold(C)$ che approssima $bold(A)$ secondo altri criteri? Certamente si potrebbe fare, ma una proprietà che distingue $bold(B)$ è che può essere riformulata al limite in quest'equazione di tipo Bolztmann in cui l'indice $I$ nel microstato è diventato il grado $K$.
    ][
      Per comprendere a pieno il significato alquanto teorico di tale trasformazione è meglio rappresentarla graficamente: di fatto si parte da una topologia esatta, descritta da $bold(A)$, quindi mediante $bold(B)$ si arriva a una approssimata (per semplicità visiva il grafo è indiretto anziché diretto) avente però lo stesso microstato, infine si giunge a una nella quale non è piú possibile distinguere i singoli nodi perché il microstato perde l'indice in luogo del grado: si definisce dunque una classe di grafi tutti con uguale distribuzione dei gradi, rilassando in tal modo la topologia. Ciò permette di avvicinarsi di piú al contesto particellare originale da cui è nata l'equazione di tipo Boltzmann.
    ][
      Con tutto ciò si può parlare dei risultati. Innanzitutto sono stati ottenuti attraverso questo metodo di Monte Carlo che è la scelta tipica nella teoria cinetica dei sistemi multiagente siccome, rispetto ad altri metodi, è quello piú fisico siccome permette di recuperare la fisica particellare soggiacente all'equazione di Boltzmann. Nel dettaglio si parte da un vettore iniziale $pzclS^0$, uniforme rispetto alla popolazione totale $P$, quindi per ogni passo temporale si permutano indipendentemente i primi $N$ indici, poi si accoppiano i primi $N\/2$ agenti coi restanti, quindi se l'interazione avviene ($Theta=1$) si modificano le taglie seguendo esattamente l'$[text("AR")]_text("S")$, altrimenti si lasciano invariate. Infine l'istogramma del vettore in ogn'istante approssima la vera distribuzione. È anche ovvio che, data la natura stocastica dell'algoritmo, sia necessario considerare piú simulazioni per ottenere un risultato statisticamente significativo.
    ][
      I primi risultati sono stati ottenuti tramite questa regola d'emigrazione ispirata da una funzione di Hill di ordine $alpha$: in poche parole descrive come il tasso d'emigrazione è maggiore verso città piú popolate. Sono state studiate due configurazioni: una è risultata stabile ma banale, nel senso che la distribuzione rimane intorno a quella iniziale uniforme, mentre l'altra è instabile ed è quella mostrata nei grafici. I primi due confermano che esistono delle classi di gradi che si spopolano sia esattamente che approsimatamente.
    ][
      Invece la distribuzione conferma quanto detto mostrando che vi sono delle realizzazioni nelle quali alcune città raggiungono una popolazione di $10^(-50)$, mentre le dispersioni identificano la classe spopolata coincidente coi centri medi secondo i gradi. L'unico lato positivo è che le simulazioni esatta e approssimata corrispondono abbastanza bene ma nel complesso si può affermare che il fenomeno della migrazione molto probabilmente non dipende _solo_ dal fattore della popolazione relativa.
    ][
      Visti i pessimi risultati si può pensare di modificare la precedente regola aggiungendo anche il grado relativo: cosí ragionando il tasso d'emigrazione è maggiore verso città sia piú popolate che piú connesse. In effetti da questa configurazione studiata i risultati sono già migliori perché convergono non mostrando segni di spopolamento.
    ][
      Invece le distribuzioni approssimano bene quelle reali, mentre le dispersioni non riescono a cogliere la correlazione positiva di quella reale assomigliando di piú a una relazione biunivoca tra popolazione e grado. Confrondando le due simulazioni, si può d'altra parte vedere come quelle esatta e approssimata siano quas'identiche.
    ][
      Si può allora analizzare l'indice di Pareto: quelli approssimato ed esatto sono tra di loro quas'identici, ed entrambi sono coerenti con quello reale differenziandosi solo al second'ordine. D'altra canto, confrontando Pareto e bilognormale, si può notare in tutt'i casi come la seconda segua meglio la coda rispetto alla prima sebbene nella prima metà coincidano. Dunque da questi risultati si può dire che il fenomeno della migrazione molto probabilmente dipende da _almeno_ questi due fattori relativi.
    ][
      È rilevante sottolineare che la convergenza della #BracketRef("RE","SK") non è scontata. Per mostrarlo si considerino i due nodi in figura _in condizioni stazionarie_ ove quello a sinistra è relativamente maggiore rispetto a quello a destra; di conseguenza è chiaro che s'instaura un flusso accentrante da $i_*$ verso $i$, ma, essendo la configurazione stabile, deve anche sussistere un flusso inverso che stabilizza il precedente; eppure, confrontando il valore della funzione di Hill, ci si rende conto che è minore colla nuova regola d'emigrazione, ovvero che il retroflusso è piú piccolo, e quindi ci si aspetterebbe piú instabile. Cionnonostante converge lasciando questo comportamento una domanda teorica aperta.
      // Eppure, a prescindere dai buoni risultati, ci si potrebbe chiedere se aggiungere il grado relativo sia una scelta modellisticamente sensata oppure una forzatura del modello a favorire città piú connesse.
    ][
      Tuttavia il grado è un informazione puramente binaria: è la somma della presenza o assenza di tutt'i lati di un nodo, e non riesce cosí a cogliere effettivamente l'uso di una rete di trasporto; sostituendo, allora, quest'ultimo col numero totale di pendolari $w$ verso una città, si riescono a distinguire due città con ugual grado ma diverso traffico interurbano. Da questa configurazione analogamente a prima i risultati convergono, ...
    ][
      ... le distribuzioni corrispondono quasi perfettamente a quella reale e lo stesso si può dire anche per le dispersioni che mostrano una correlazione sempre positiva ma meno stretta del caso precedente e piú in linea con quella reale. Queste ultime non sono però perfette perché, seppure lievemente, sovrastimano le taglie dei centri medio-minori e sottostimano quelle dei centri maggiori; ciò è molto probabilmente attribuibile alla mancanza del traffico _intraurbano_ nella definizione della forza. Invece tra la simulazione esatta e approssimata anche in questo caso non vi sono differenze significative.
    ][
      Sulla Pareto e la bilognormale valgono gli stessi commenti di prima: tutt'i gl'indici, approssimato, esatto e reale, si differenziano tra di loro solo al second'ordine, da cui segue la loro somiglianza; anche in questo caso la bilognormale segue meglio la coda della distribuzione, seppure coincida colla Pareto nella prima metà. Da questi risultati è chiaro che il fenomeno della migrazione molto probabilmente è governato da questi due fattori relativi.
    ][
      È anche possibile interpretare i due parametri principali che caratterizzano tutte le regole finora viste tramite due studi parametrici che, ovviamente per brevità, non sono mostrati. Innanzitutto $lambda$ è l'attrazione esercitata dai centri rel. maggiori: modifica leggermente il comportamento medio e un suo aumento allunga il tempo di convergenza; viceversa $alpha$ è la repulsione sprigionata dai centri rel. minori: la sua modifica ha profondi ripercussioni su ogni aspetto dei risultati, specialmente per l'indice di Pareto; è anche molto probabilmente inversamente legata al retroflusso stabilizzante: piú grade è $alpha$ minore è il retroflusso e quindi piú instabile è il modello. In effetti $lambda$ è un semplice riscalamento, influenzando maggiormente la parte inferiore ($r>1$, verso i centri rel. maggiori), mentre $alpha$ altera l'intera forma, soprattutto la parte inferiore ($r<=1$, verso i centri rel. minori).
    ][
      Il lavoro qui presentato non è affatto esaustivo e conduce a molteplici sviluppi futuri. Se ne elencano per esempio quattro: 1) si possono definire regole d'emigrazione piú complesse, non necessariamente di Hill, usando pure altri fattori dalla teoria dei grafi; 2) si può tentare di ricavare l'equazione di Fokker-Planck per studiare sia la distribuzione stazionaria che il retroflusso stabilizzante; 3) si può considerare un grafo dinamico coevolvente colla distribuzione della popolazione; infine, 4) si può verificare questa teoria in altri contesti europei o internazionali.
    ][
      Grazie dell'attenzione.
    ]
//#endregion
