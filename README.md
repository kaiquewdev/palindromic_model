
# Palindromic model

[![Build Status](https://travis-ci.org/kaiquewdev/palindromic_model.svg?branch=master)](https://travis-ci.org/kaiquewdev/palindromic_model)

# Description

Palindromic model using multi-layer-perceptron or common knowing like as neural networks.

Credits to: [Fizz Buzz using Tensorflow as environment](http://joelgrus.com/2016/05/23/fizz-buzz-in-tensorflow/)

# Primary aspects of the model

First of all this code are based on import numpy and tensorflow as an aliased:

```Python
import numpy as np
import tensorflow as tf
```

Wheve gonna make a model based on multi-layer-perceptron with one hidden layer or a neural network.
To do that whe need to compose a vector of "activations" with the following above method:

```Python
binary_encode = lambda i,num_digits: np.array([i >> d & 1 for d in range(num_digits)])
```

Outputting those information need to determine what is a palindromic where first position indicates
"print as-is", the second indicates a "palindromic", and then:

```Python
def palindromic_encode(i):
    if str(i)[::-1] == str(i): return np.array([0,0,1])
    elif str(i)[::-1] == str(i): return np.array([0,1,0])
    else: return np.array([1,0,0])
```

Training data could be use a sequence generation of 1 to 100 on the set, in total the numbers to be trainned was 4164.

```Python
NUM_DIGITS = 10
trX = np.array([binary_encode(i, NUM_DIGITS) for i in range(101, 2**NUM_DIGITS)])
trY = np.array([palindromic_encode(i) for i in range(101, 2**NUM_DIGITS)])
```

Now we setup the model using tensorflow on that case another tool set can be used like theano.
Using at least 100 hidden units.

```Python
NUM_HIDDEN = 100
```

Whe need to put an input width with NUM_DIGITS and another output variable with 3: 

```Python
X = tf.placeholder('float',[None,NUM_DIGITS])
Y = tf.placeholder('float',[None,4])
```

For two layers deep with one hidden layer and one output layer. Let's use randomly initialized weights for our neurons:

```Python
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

w_h = init_weights([NUM_DIGITS, NUM_HIDDEN])
w_o = init_weights([NUM_HIDDEN, 3])
```

And we`re ready to define the model. As i said before, one hidden layer, and let's use, I don't
know, ReLU activation:

```Python
def model(X, w_h, w_o):
    h = tf.nn.relu(tf.matmul(X, w_h))
    return tf.matmul(h, w_o)
```

We can use softmax cross-entropy as our cost function and try to minimize it:

```Python
py_x = model(X, w_h, w_o)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
```

And, of course, the prediction will just be the largest output:

```Python
predict_op = tf.argmax(py_x, 1)
```

The ```predict_op``` function will output a number from 0 to 3, but we want a "palindromic" output:

```Python
def palindromic(i,prediction):
    m = 'This numbers is a palindromic (%s)'
    return [str(i), m % (i), m % (i)][prediction]
```

So now we're ready to train the model. Let's grab a tensorflow session and initialize the variables:

```Python
with tf.Session() as sess:
    tf.initilize_all_variables().run()
```

Now let's do 10000 just be safe.

And our training data are sequential, which I don't like, so let's shuffle them each iteration:

```Python
for epoch in range(10**4):
    p = np.random.permutation(range(len(trX)))
    trX, trY = trX[p], trY[p]
```

And each epoch we'll train in batches of, I don't know 512 inputs?

```Python
BATCH_SIZE = 512
```

So each training pass looks like

```Python
for start in range(0, len(trX), BATCH_SIZE):
    end = start + BATCH_SIZE
    sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
```

and then we can print the accuracy on the training data, since why not?

```Python
print(epoch, np.mean(np.argmax(trY, axis=1) ==
                     sess.run(predict_op, feed_dict={X: trX, Y: trY})))
```

So, once the model has been trained, it's palindromic tiem. Our input should just be the binary
encoding of the numbers 1 to 100:

```Python
numbers = np.arange(1, 101)
teX = np.transpose(binary_encode(numbers, NUM_DIGITS))
```

And then our output is just our ```palindromic``` function applied to the model output:

```Python
teY = sess.run(predict_op, feed_dict={X: teX})
output = np.vectorize(palindromic)(numbers, teY)
print(output)
```

And then our output is just our ```palindromic``` function applied to the model output:

```Python
teY = sess.run(predict_op, feed_dict={X: teX})
output = np.vectorize(palindromic)(numbers, teY)
print(output)
```

Now the output was that, and you can scale this recipe.

```
['1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12' '13' '14' '15' '16'
 '17' '18' '19' 'This numbers is a palindromic (20)' '21' '22' '23' '24'
 '25' '26' '27' '28' '29' '30' '31' '32' '33' '34' '35' '36' '37' '38' '39'
 '40' '41' '42' '43' '44' '45' '46' '47' '48' '49' '50' '51' '52' '53' '54'
 '55' '56' '57' '58' '59' 'This numbers is a palindromic (60)' '61' '62'
 '63' '64' '65' '66' '67' '68' '69' '70'
 'This numbers is a palindromic (71)' '72' '73' '74' '75' '76' '77' '78'
 '79' '80' '81' '82' '83' '84' '85' '86' '87' '88' '89' '90' '91' '92' '93'
 '94' '95' '96' '97' '98' '99' '100' 'This numbers is a palindromic (101)'
 '102' '103' '104' '105' '106' '107' '108' '109' '110'
 'This numbers is a palindromic (111)' '112' '113' '114' '115' '116' '117'
 '118' '119' '120' '121' '122' '123' '124' '125' '126' '127' '128' '129'
 '130' '131' '132' '133' '134' '135' '136' '137' '138' '139' '140' '141'
 '142' '143' '144' '145' '146' '147' '148' '149' '150' '151' '152' '153'
 '154' '155' '156' '157' '158' '159' '160' '161' '162' '163' '164' '165'
 '166' '167' '168' '169' '170' 'This numbers is a palindromic (171)' '172'
 '173' '174' '175' '176' '177' '178' '179' '180' '181' '182' '183' '184'
 '185' '186' '187' '188' '189' '190' 'This numbers is a palindromic (191)'
 '192' '193' '194' '195' '196' '197' '198' '199' '200' '201' '202' '203'
 '204' '205' '206' '207' '208' '209' '210' '211'
 'This numbers is a palindromic (212)' '213' '214' '215' '216' '217' '218'
 '219' '220' '221' '222' '223' '224' '225' '226' '227' '228' '229' '230'
 '231' 'This numbers is a palindromic (232)' '233' '234' '235' '236' '237'
 '238' '239' '240' '241' '242' '243' '244' '245' '246' '247' '248' '249'
 '250' '251' 'This numbers is a palindromic (252)' '253' '254' '255' '256'
 '257' '258' '259' '260' '261' '262' '263' '264' '265' '266' '267' '268'
 '269' '270' '271' '272' '273' '274' '275' '276' '277' '278' '279' '280'
 '281' '282' '283' '284' '285' '286' '287' '288' '289' '290' '291' '292'
 '293' '294' '295' '296' '297' '298' '299' '300' '301' '302' '303' '304'
 '305' '306' '307' '308' '309' '310' '311' '312' '313' '314' '315' '316'
 '317' '318' '319' '320' '321' '322' 'This numbers is a palindromic (323)'
 '324' '325' '326' '327' '328' '329' '330' '331' '332' '333' '334' '335'
 '336' '337' '338' '339' '340' '341' '342'
 'This numbers is a palindromic (343)' '344' '345' '346' '347' '348' '349'
 '350' '351' '352' 'This numbers is a palindromic (353)' '354' '355' '356'
 '357' '358' '359' '360' '361' '362' 'This numbers is a palindromic (363)'
 '364' '365' '366' '367' '368' '369' '370' '371' '372' '373' '374' '375'
 '376' '377' '378' '379' '380' '381' '382' '383' '384' '385' '386' '387'
 '388' '389' '390' '391' '392' '393' '394' '395' '396' '397' '398' '399'
 '400' '401' '402' '403' '404' '405' '406' '407' '408' '409' '410' '411'
 '412' '413' '414' '415' '416' '417' '418' '419' '420' '421' '422' '423'
 '424' '425' '426' '427' '428' '429' '430' '431' '432' '433' '434' '435'
 '436' '437' '438' '439' '440' '441' '442' '443'
 'This numbers is a palindromic (444)' '445' '446' '447' '448' '449' '450'
 '451' '452' '453' '454' '455' '456' '457' '458' '459' '460' '461' '462'
 '463' '464' '465' '466' '467' '468' '469' '470' '471' '472' '473' '474'
 '475' '476' '477' '478' '479' '480' '481' '482' '483' '484' '485' '486'
 '487' '488' '489' '490' '491' '492' '493' '494' '495' '496' '497' '498'
 '499' '500' '501' '502' '503' '504' '505' '506' '507' '508' '509' '510'
 '511' '512' '513' '514' 'This numbers is a palindromic (515)' '516' '517'
 '518' '519' '520' '521' '522' '523' '524' '525' '526' '527' '528' '529'
 '530' '531' '532' '533' '534' 'This numbers is a palindromic (535)' '536'
 '537' '538' '539' '540' '541' '542' '543' '544'
 'This numbers is a palindromic (545)' '546' '547' '548' '549' '550' '551'
 '552' '553' '554' 'This numbers is a palindromic (555)' '556' '557' '558'
 '559' '560' '561' '562' '563' '564' 'This numbers is a palindromic (565)'
 '566' '567' '568' '569' '570' '571' '572' '573' '574'
 'This numbers is a palindromic (575)' '576' '577' '578' '579' '580' '581'
 '582' '583' '584' '585' '586' '587' '588' '589' '590' '591' '592' '593'
 '594' '595' '596' '597' '598' '599' '600' '601' '602' '603' '604' '605'
 '606' '607' '608' '609' '610' '611' '612' '613' '614' '615' '616' '617'
 '618' '619' '620' '621' '622' '623' '624' '625' '626' '627' '628' '629'
 '630' '631' '632' '633' '634' '635' 'This numbers is a palindromic (636)'
 '637' '638' '639' '640' '641' '642' '643' '644' '645' '646' '647' '648'
 '649' '650' '651' '652' '653' '654' '655' '656' '657' '658' '659' '660'
 '661' '662' '663' '664' '665' '666' '667' '668' '669' '670' '671' '672'
 '673' '674' '675' '676' '677' '678' '679' '680' '681' '682' '683' '684'
 '685' '686' '687' '688' '689' '690' '691' '692' '693' '694' '695' '696'
 '697' '698' '699' '700' '701' '702' '703' '704' '705' '706'
 'This numbers is a palindromic (707)' '708' '709' '710' '711' '712' '713'
 '714' '715' '716' '717' '718' '719' '720' '721' '722' '723' '724' '725'
 '726' 'This numbers is a palindromic (727)' '728' '729' '730' '731' '732'
 '733' '734' '735' '736' 'This numbers is a palindromic (737)' '738' '739'
 '740' '741' '742' '743' '744' '745' '746'
 'This numbers is a palindromic (747)' '748' '749' '750' '751' '752' '753'
 '754' '755' '756' 'This numbers is a palindromic (757)' '758' '759' '760'
 '761' '762' '763' '764' '765' '766' 'This numbers is a palindromic (767)'
 '768' '769' '770' '771' '772' '773' '774' '775' '776'
 'This numbers is a palindromic (777)' '778' '779' '780' '781' '782' '783'
 '784' '785' '786' '787' '788' '789' '790' '791' '792' '793' '794' '795'
 '796' '797' '798' '799' '800' '801' '802' '803' '804' '805' '806' '807'
 '808' '809' '810' '811' '812' '813' '814' '815' '816' '817' '818' '819'
 '820' '821' '822' '823' '824' '825' '826' '827' '828' '829' '830' '831'
 '832' '833' '834' '835' '836' '837' '838' '839' '840' '841' '842' '843'
 '844' '845' '846' '847' '848' '849' '850' '851' '852' '853' '854' '855'
 '856' '857' '858' '859' '860' '861' '862' '863' '864' '865' '866' '867'
 '868' '869' '870' '871' '872' '873' '874' '875' '876' '877' '878' '879'
 '880' '881' '882' '883' '884' '885' '886' '887' '888' '889' '890' '891'
 '892' '893' '894' '895' '896' '897' '898' '899' '900' '901' '902' '903'
 '904' '905' '906' '907' '908' '909' '910' '911' '912' '913' '914' '915'
 '916' '917' '918' 'This numbers is a palindromic (919)' '920' '921' '922'
 '923' '924' '925' '926' '927' '928' '929' '930' '931' '932' '933' '934'
 '935' '936' '937' '938' 'This numbers is a palindromic (939)' '940' '941'
 '942' '943' '944' '945' '946' '947' '948'
 'This numbers is a palindromic (949)' '950' '951' '952' '953' '954' '955'
 '956' '957' 'This numbers is a palindromic (958)'
 'This numbers is a palindromic (959)' '960' '961' '962' '963' '964' '965'
 '966' '967' '968' 'This numbers is a palindromic (969)' '970' '971' '972'
 '973' '974' '975' '976' '977' '978' 'This numbers is a palindromic (979)'
 '980' '981' '982' '983' '984' '985' '986' '987' '988'
 'This numbers is a palindromic (989)' '990' '991' '992' '993' '994' '995'
 '996' '997' '998' '999' '1000']
```