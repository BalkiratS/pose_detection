бЛ
Ъ¤
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
Z
BroadcastTo

input"T
shape"Tidx
output"T"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
A
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
о
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
М
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu6
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
-
Sqrt
x"T
y"T"
Ttype:

2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
┴
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ии
@
StaticRegexFullMatch	
input

output
"
patternstring
ў
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
М
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.11.02unknown8╡┴

^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
~
Adam/v/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/v/dense_2/bias
w
'Adam/v/dense_2/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_2/bias*
_output_shapes
:*
dtype0
~
Adam/m/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/m/dense_2/bias
w
'Adam/m/dense_2/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_2/bias*
_output_shapes
:*
dtype0
Ж
Adam/v/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*&
shared_nameAdam/v/dense_2/kernel

)Adam/v/dense_2/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_2/kernel*
_output_shapes

:(*
dtype0
Ж
Adam/m/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*&
shared_nameAdam/m/dense_2/kernel

)Adam/m/dense_2/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_2/kernel*
_output_shapes

:(*
dtype0
~
Adam/v/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*$
shared_nameAdam/v/dense_1/bias
w
'Adam/v/dense_1/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_1/bias*
_output_shapes
:(*
dtype0
~
Adam/m/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*$
shared_nameAdam/m/dense_1/bias
w
'Adam/m/dense_1/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_1/bias*
_output_shapes
:(*
dtype0
Ж
Adam/v/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z(*&
shared_nameAdam/v/dense_1/kernel

)Adam/v/dense_1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_1/kernel*
_output_shapes

:Z(*
dtype0
Ж
Adam/m/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z(*&
shared_nameAdam/m/dense_1/kernel

)Adam/m/dense_1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_1/kernel*
_output_shapes

:Z(*
dtype0
z
Adam/v/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*"
shared_nameAdam/v/dense/bias
s
%Adam/v/dense/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense/bias*
_output_shapes
:Z*
dtype0
z
Adam/m/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*"
shared_nameAdam/m/dense/bias
s
%Adam/m/dense/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense/bias*
_output_shapes
:Z*
dtype0
В
Adam/v/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*$
shared_nameAdam/v/dense/kernel
{
'Adam/v/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense/kernel*
_output_shapes

:Z*
dtype0
В
Adam/m/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*$
shared_nameAdam/m/dense/kernel
{
'Adam/m/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense/kernel*
_output_shapes

:Z*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:(*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:(*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z(*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:Z(*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:Z*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:Z*
dtype0
z
serving_default_input_1Placeholder*'
_output_shapes
:         '*
dtype0*
shape:         '
Т
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *+
f&R$
"__inference_signature_wrapper_7180

NoOpNoOp
о?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*щ>
value▀>B▄> B╒>
Ж
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer-19
layer-20
layer-21
layer-22
layer-23
layer-24
layer-25
layer-26
layer-27
layer-28
layer-29
layer-30
 layer-31
!layer-32
"layer-33
#layer-34
$layer-35
%layer-36
&layer-37
'layer-38
(layer-39
)layer-40
*layer-41
+layer_with_weights-0
+layer-42
,layer_with_weights-1
,layer-43
-layer_with_weights-2
-layer-44
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses
4_default_save_signature
5	optimizer
6
signatures*
* 
О
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses* 

=	keras_api* 

>	keras_api* 

?	keras_api* 

@	keras_api* 

A	keras_api* 

B	keras_api* 

C	keras_api* 

D	keras_api* 

E	keras_api* 

F	keras_api* 

G	keras_api* 

H	keras_api* 

I	keras_api* 

J	keras_api* 

K	keras_api* 

L	keras_api* 

M	keras_api* 

N	keras_api* 

O	keras_api* 

P	keras_api* 

Q	keras_api* 

R	keras_api* 

S	keras_api* 

T	keras_api* 

U	keras_api* 

V	keras_api* 

W	keras_api* 

X	keras_api* 

Y	keras_api* 

Z	keras_api* 

[	keras_api* 

\	keras_api* 

]	keras_api* 

^	keras_api* 

_	keras_api* 

`	keras_api* 

a	keras_api* 

b	keras_api* 

c	keras_api* 
О
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses* 
ж
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses

pkernel
qbias*
ж
r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses

xkernel
ybias*
и
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses
Аkernel
	Бbias*
0
p0
q1
x2
y3
А4
Б5*
0
p0
q1
x2
y3
А4
Б5*
* 
╡
Вnon_trainable_variables
Гlayers
Дmetrics
 Еlayer_regularization_losses
Жlayer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
4_default_save_signature
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses*
:
Зtrace_0
Иtrace_1
Йtrace_2
Кtrace_3* 
:
Лtrace_0
Мtrace_1
Нtrace_2
Оtrace_3* 
* 
И
П
_variables
Р_iterations
С_learning_rate
Т_index_dict
У
_momentums
Ф_velocities
Х_update_step_xla*

Цserving_default* 
* 
* 
* 
Ц
Чnon_trainable_variables
Шlayers
Щmetrics
 Ъlayer_regularization_losses
Ыlayer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses* 

Ьtrace_0* 

Эtrace_0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
Ц
Юnon_trainable_variables
Яlayers
аmetrics
 бlayer_regularization_losses
вlayer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses* 

гtrace_0* 

дtrace_0* 

p0
q1*

p0
q1*
* 
Ш
еnon_trainable_variables
жlayers
зmetrics
 иlayer_regularization_losses
йlayer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses*

кtrace_0* 

лtrace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

x0
y1*

x0
y1*
* 
Ш
мnon_trainable_variables
нlayers
оmetrics
 пlayer_regularization_losses
░layer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses*

▒trace_0* 

▓trace_0* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

А0
Б1*

А0
Б1*
* 
Ш
│non_trainable_variables
┤layers
╡metrics
 ╢layer_regularization_losses
╖layer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

╕trace_0* 

╣trace_0* 
^X
VARIABLE_VALUEdense_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
т
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
+42
,43
-44*

║0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
o
Р0
╗1
╝2
╜3
╛4
┐5
└6
┴7
┬8
├9
─10
┼11
╞12*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
4
╗0
╜1
┐2
┴3
├4
┼5*
4
╝0
╛1
└2
┬3
─4
╞5*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
╟	variables
╚	keras_api

╔total

╩count*
^X
VARIABLE_VALUEAdam/m/dense/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/dense/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/m/dense/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/v/dense/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_1/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_1/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/dense_1/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/dense_1/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_2/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_2/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_2/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_2/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*

╔0
╩1*

╟	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
х
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp'Adam/m/dense/kernel/Read/ReadVariableOp'Adam/v/dense/kernel/Read/ReadVariableOp%Adam/m/dense/bias/Read/ReadVariableOp%Adam/v/dense/bias/Read/ReadVariableOp)Adam/m/dense_1/kernel/Read/ReadVariableOp)Adam/v/dense_1/kernel/Read/ReadVariableOp'Adam/m/dense_1/bias/Read/ReadVariableOp'Adam/v/dense_1/bias/Read/ReadVariableOp)Adam/m/dense_2/kernel/Read/ReadVariableOp)Adam/v/dense_2/kernel/Read/ReadVariableOp'Adam/m/dense_2/bias/Read/ReadVariableOp'Adam/v/dense_2/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*#
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *&
f!R
__inference__traced_save_7640
и
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias	iterationlearning_rateAdam/m/dense/kernelAdam/v/dense/kernelAdam/m/dense/biasAdam/v/dense/biasAdam/m/dense_1/kernelAdam/v/dense_1/kernelAdam/m/dense_1/biasAdam/v/dense_1/biasAdam/m/dense_2/kernelAdam/v/dense_2/kernelAdam/m/dense_2/biasAdam/v/dense_2/biastotalcount*"
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *)
f$R"
 __inference__traced_restore_7716в╞	
Ы

Є
A__inference_dense_1_layer_call_and_return_conditional_losses_6700

inputs0
matmul_readvariableop_resource:Z(-
biasadd_readvariableop_resource:(
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Z(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         (r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         (R
Relu6Relu6BiasAdd:output:0*
T0*'
_output_shapes
:         (b
IdentityIdentityRelu6:activations:0^NoOp*
T0*'
_output_shapes
:         (w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         Z: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         Z
 
_user_specified_nameinputs
р
¤
$__inference_model_layer_call_fn_7197

inputs
unknown:Z
	unknown_0:Z
	unknown_1:Z(
	unknown_2:(
	unknown_3:(
	unknown_4:
identityИвStatefulPartitionedCallИ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_6724o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         ': : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         '
 
_user_specified_nameinputs
╝
У
&__inference_dense_2_layer_call_fn_7540

inputs
unknown:(
	unknown_0:
identityИвStatefulPartitionedCall╓
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_6717o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         (: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         (
 
_user_specified_nameinputs
╕
С
$__inference_dense_layer_call_fn_7500

inputs
unknown:Z
	unknown_0:Z
identityИвStatefulPartitionedCall╘
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         Z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_6683o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Z`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Яy
Є
?__inference_model_layer_call_and_return_conditional_losses_6724

inputs

dense_6684:Z

dense_6686:Z
dense_1_6701:Z(
dense_1_6703:(
dense_2_6718:(
dense_2_6720:
identityИвdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCallвdense_2/StatefulPartitionedCall╝
reshape_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_reshape_1_layer_call_and_return_conditional_losses_6574Б
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            Г
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           Г
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ш
&tf.__operators__.getitem/strided_sliceStridedSlice"reshape_1/PartitionedCall:output:05tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         *

begin_mask*
end_maskh
&tf.compat.v1.gather_1/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :П
tf.compat.v1.gather_1/GatherV2GatherV2/tf.__operators__.getitem/strided_slice:output:0/tf.compat.v1.gather_1/GatherV2/indices:output:0,tf.compat.v1.gather_1/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         f
$tf.compat.v1.gather/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :c
!tf.compat.v1.gather/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :Й
tf.compat.v1.gather/GatherV2GatherV2/tf.__operators__.getitem/strided_slice:output:0-tf.compat.v1.gather/GatherV2/indices:output:0*tf.compat.v1.gather/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         [
tf.math.multiply/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Х
tf.math.multiply/MulMul%tf.compat.v1.gather/GatherV2:output:0tf.math.multiply/Mul/y:output:0*
T0*'
_output_shapes
:         ]
tf.math.multiply_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ы
tf.math.multiply_1/MulMul'tf.compat.v1.gather_1/GatherV2:output:0!tf.math.multiply_1/Mul/y:output:0*
T0*'
_output_shapes
:         p
tf.compat.v1.size/SizeSize/tf.__operators__.getitem/strided_slice:output:0*
T0*
_output_shapes
: Л
tf.__operators__.add/AddV2AddV2tf.math.multiply/Mul:z:0tf.math.multiply_1/Mul:z:0*
T0*'
_output_shapes
:         _
tf.expand_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :е
tf.expand_dims/ExpandDims
ExpandDimstf.__operators__.add/AddV2:z:0&tf.expand_dims/ExpandDims/dim:output:0*
T0*+
_output_shapes
:         c
!tf.compat.v1.floor_div/FloorDiv/yConst*
_output_shapes
: *
dtype0*
value	B :Щ
tf.compat.v1.floor_div/FloorDivFloorDivtf.compat.v1.size/Size:output:0*tf.compat.v1.floor_div/FloorDiv/y:output:0*
T0*
_output_shapes
: e
#tf.broadcast_to/BroadcastTo/shape/1Const*
_output_shapes
: *
dtype0*
value	B :e
#tf.broadcast_to/BroadcastTo/shape/2Const*
_output_shapes
: *
dtype0*
value	B :╪
!tf.broadcast_to/BroadcastTo/shapePack#tf.compat.v1.floor_div/FloorDiv:z:0,tf.broadcast_to/BroadcastTo/shape/1:output:0,tf.broadcast_to/BroadcastTo/shape/2:output:0*
N*
T0*
_output_shapes
:░
tf.broadcast_to/BroadcastToBroadcastTo"tf.expand_dims/ExpandDims:output:0*tf.broadcast_to/BroadcastTo/shape:output:0*
T0*+
_output_shapes
:         и
tf.math.subtract/SubSub/tf.__operators__.getitem/strided_slice:output:0$tf.broadcast_to/BroadcastTo:output:0*
T0*+
_output_shapes
:         h
&tf.compat.v1.gather_7/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_7/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :°
tf.compat.v1.gather_7/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_7/GatherV2/indices:output:0,tf.compat.v1.gather_7/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         h
&tf.compat.v1.gather_6/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_6/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :°
tf.compat.v1.gather_6/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_6/GatherV2/indices:output:0,tf.compat.v1.gather_6/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         ]
tf.math.multiply_6/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ы
tf.math.multiply_6/MulMul'tf.compat.v1.gather_6/GatherV2:output:0!tf.math.multiply_6/Mul/y:output:0*
T0*'
_output_shapes
:         ]
tf.math.multiply_7/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ы
tf.math.multiply_7/MulMul'tf.compat.v1.gather_7/GatherV2:output:0!tf.math.multiply_7/Mul/y:output:0*
T0*'
_output_shapes
:         [
tf.compat.v1.size_1/SizeSizetf.math.subtract/Sub:z:0*
T0*
_output_shapes
: П
tf.__operators__.add_3/AddV2AddV2tf.math.multiply_6/Mul:z:0tf.math.multiply_7/Mul:z:0*
T0*'
_output_shapes
:         a
tf.expand_dims_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :л
tf.expand_dims_1/ExpandDims
ExpandDims tf.__operators__.add_3/AddV2:z:0(tf.expand_dims_1/ExpandDims/dim:output:0*
T0*+
_output_shapes
:         e
#tf.compat.v1.floor_div_1/FloorDiv/yConst*
_output_shapes
: *
dtype0*
value	B :Я
!tf.compat.v1.floor_div_1/FloorDivFloorDiv!tf.compat.v1.size_1/Size:output:0,tf.compat.v1.floor_div_1/FloorDiv/y:output:0*
T0*
_output_shapes
: h
&tf.compat.v1.gather_3/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_3/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :°
tf.compat.v1.gather_3/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_3/GatherV2/indices:output:0,tf.compat.v1.gather_3/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         h
&tf.compat.v1.gather_2/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_2/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :°
tf.compat.v1.gather_2/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_2/GatherV2/indices:output:0,tf.compat.v1.gather_2/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         h
&tf.compat.v1.gather_5/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_5/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :°
tf.compat.v1.gather_5/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_5/GatherV2/indices:output:0,tf.compat.v1.gather_5/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         h
&tf.compat.v1.gather_4/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_4/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :°
tf.compat.v1.gather_4/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_4/GatherV2/indices:output:0,tf.compat.v1.gather_4/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         g
%tf.broadcast_to_1/BroadcastTo/shape/1Const*
_output_shapes
: *
dtype0*
value	B :g
%tf.broadcast_to_1/BroadcastTo/shape/2Const*
_output_shapes
: *
dtype0*
value	B :р
#tf.broadcast_to_1/BroadcastTo/shapePack%tf.compat.v1.floor_div_1/FloorDiv:z:0.tf.broadcast_to_1/BroadcastTo/shape/1:output:0.tf.broadcast_to_1/BroadcastTo/shape/2:output:0*
N*
T0*
_output_shapes
:╢
tf.broadcast_to_1/BroadcastToBroadcastTo$tf.expand_dims_1/ExpandDims:output:0,tf.broadcast_to_1/BroadcastTo/shape:output:0*
T0*+
_output_shapes
:         ]
tf.math.multiply_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ы
tf.math.multiply_2/MulMul'tf.compat.v1.gather_2/GatherV2:output:0!tf.math.multiply_2/Mul/y:output:0*
T0*'
_output_shapes
:         ]
tf.math.multiply_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ы
tf.math.multiply_3/MulMul'tf.compat.v1.gather_3/GatherV2:output:0!tf.math.multiply_3/Mul/y:output:0*
T0*'
_output_shapes
:         ]
tf.math.multiply_4/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ы
tf.math.multiply_4/MulMul'tf.compat.v1.gather_4/GatherV2:output:0!tf.math.multiply_4/Mul/y:output:0*
T0*'
_output_shapes
:         ]
tf.math.multiply_5/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ы
tf.math.multiply_5/MulMul'tf.compat.v1.gather_5/GatherV2:output:0!tf.math.multiply_5/Mul/y:output:0*
T0*'
_output_shapes
:         Х
tf.math.subtract_2/SubSubtf.math.subtract/Sub:z:0&tf.broadcast_to_1/BroadcastTo:output:0*
T0*+
_output_shapes
:         П
tf.__operators__.add_2/AddV2AddV2tf.math.multiply_4/Mul:z:0tf.math.multiply_5/Mul:z:0*
T0*'
_output_shapes
:         П
tf.__operators__.add_1/AddV2AddV2tf.math.multiply_2/Mul:z:0tf.math.multiply_3/Mul:z:0*
T0*'
_output_shapes
:         h
&tf.compat.v1.gather_8/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B : e
#tf.compat.v1.gather_8/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ё
tf.compat.v1.gather_8/GatherV2GatherV2tf.math.subtract_2/Sub:z:0/tf.compat.v1.gather_8/GatherV2/indices:output:0,tf.compat.v1.gather_8/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes

:У
tf.math.subtract_1/SubSub tf.__operators__.add_2/AddV2:z:0 tf.__operators__.add_1/AddV2:z:0*
T0*'
_output_shapes
:         Ю
tf.compat.v1.norm_1/norm/mulMul'tf.compat.v1.gather_8/GatherV2:output:0'tf.compat.v1.gather_8/GatherV2:output:0*
T0*
_output_shapes

:x
.tf.compat.v1.norm_1/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ╕
tf.compat.v1.norm_1/norm/SumSum tf.compat.v1.norm_1/norm/mul:z:07tf.compat.v1.norm_1/norm/Sum/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(u
tf.compat.v1.norm_1/norm/SqrtSqrt%tf.compat.v1.norm_1/norm/Sum:output:0*
T0*
_output_shapes

:К
 tf.compat.v1.norm_1/norm/SqueezeSqueeze!tf.compat.v1.norm_1/norm/Sqrt:y:0*
T0*
_output_shapes
:*
squeeze_dims
 Л
tf.compat.v1.norm/norm/mulMultf.math.subtract_1/Sub:z:0tf.math.subtract_1/Sub:z:0*
T0*'
_output_shapes
:         m
tf.compat.v1.norm/norm/ConstConst*
_output_shapes
:*
dtype0*
valueB"       в
tf.compat.v1.norm/norm/SumSumtf.compat.v1.norm/norm/mul:z:0%tf.compat.v1.norm/norm/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(q
tf.compat.v1.norm/norm/SqrtSqrt#tf.compat.v1.norm/norm/Sum:output:0*
T0*
_output_shapes

:k
tf.compat.v1.norm/norm/SqueezeSqueezetf.compat.v1.norm/norm/Sqrt:y:0*
T0*
_output_shapes
: ]
tf.math.multiply_8/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @К
tf.math.multiply_8/MulMul'tf.compat.v1.norm/norm/Squeeze:output:0!tf.math.multiply_8/Mul/y:output:0*
T0*
_output_shapes
: b
tf.math.reduce_max/ConstConst*
_output_shapes
:*
dtype0*
valueB: М
tf.math.reduce_max/MaxMax)tf.compat.v1.norm_1/norm/Squeeze:output:0!tf.math.reduce_max/Const:output:0*
T0*
_output_shapes
: А
tf.math.maximum/MaximumMaximumtf.math.multiply_8/Mul:z:0tf.math.reduce_max/Max:output:0*
T0*
_output_shapes
: П
tf.math.truediv/truedivRealDivtf.math.subtract/Sub:z:0tf.math.maximum/Maximum:z:0*
T0*+
_output_shapes
:         ═
flatten_1/PartitionedCallPartitionedCalltf.math.truediv/truediv:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_6670·
dense/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0
dense_6684
dense_6686*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         Z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_6683Ж
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_6701dense_1_6703*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         (*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_6700И
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_6718dense_2_6720*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_6717w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         к
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         ': : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:O K
'
_output_shapes
:         '
 
_user_specified_nameinputs
╗
_
C__inference_flatten_1_layer_call_and_return_conditional_losses_6670

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:         X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         :S O
+
_output_shapes
:         
 
_user_specified_nameinputs
Ч

Є
A__inference_dense_2_layer_call_and_return_conditional_losses_6717

inputs0
matmul_readvariableop_resource:(-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         (: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         (
 
_user_specified_nameinputs
у
■
$__inference_model_layer_call_fn_6739
input_1
unknown:Z
	unknown_0:Z
	unknown_1:Z(
	unknown_2:(
	unknown_3:(
	unknown_4:
identityИвStatefulPartitionedCallЙ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_6724o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         ': : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:         '
!
_user_specified_name	input_1
у
■
$__inference_model_layer_call_fn_6941
input_1
unknown:Z
	unknown_0:Z
	unknown_1:Z(
	unknown_2:(
	unknown_3:(
	unknown_4:
identityИвStatefulPartitionedCallЙ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_6909o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         ': : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:         '
!
_user_specified_name	input_1
Яy
Є
?__inference_model_layer_call_and_return_conditional_losses_6909

inputs

dense_6893:Z

dense_6895:Z
dense_1_6898:Z(
dense_1_6900:(
dense_2_6903:(
dense_2_6905:
identityИвdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCallвdense_2/StatefulPartitionedCall╝
reshape_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_reshape_1_layer_call_and_return_conditional_losses_6574Б
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            Г
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           Г
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ш
&tf.__operators__.getitem/strided_sliceStridedSlice"reshape_1/PartitionedCall:output:05tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         *

begin_mask*
end_maskh
&tf.compat.v1.gather_1/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :П
tf.compat.v1.gather_1/GatherV2GatherV2/tf.__operators__.getitem/strided_slice:output:0/tf.compat.v1.gather_1/GatherV2/indices:output:0,tf.compat.v1.gather_1/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         f
$tf.compat.v1.gather/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :c
!tf.compat.v1.gather/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :Й
tf.compat.v1.gather/GatherV2GatherV2/tf.__operators__.getitem/strided_slice:output:0-tf.compat.v1.gather/GatherV2/indices:output:0*tf.compat.v1.gather/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         [
tf.math.multiply/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Х
tf.math.multiply/MulMul%tf.compat.v1.gather/GatherV2:output:0tf.math.multiply/Mul/y:output:0*
T0*'
_output_shapes
:         ]
tf.math.multiply_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ы
tf.math.multiply_1/MulMul'tf.compat.v1.gather_1/GatherV2:output:0!tf.math.multiply_1/Mul/y:output:0*
T0*'
_output_shapes
:         p
tf.compat.v1.size/SizeSize/tf.__operators__.getitem/strided_slice:output:0*
T0*
_output_shapes
: Л
tf.__operators__.add/AddV2AddV2tf.math.multiply/Mul:z:0tf.math.multiply_1/Mul:z:0*
T0*'
_output_shapes
:         _
tf.expand_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :е
tf.expand_dims/ExpandDims
ExpandDimstf.__operators__.add/AddV2:z:0&tf.expand_dims/ExpandDims/dim:output:0*
T0*+
_output_shapes
:         c
!tf.compat.v1.floor_div/FloorDiv/yConst*
_output_shapes
: *
dtype0*
value	B :Щ
tf.compat.v1.floor_div/FloorDivFloorDivtf.compat.v1.size/Size:output:0*tf.compat.v1.floor_div/FloorDiv/y:output:0*
T0*
_output_shapes
: e
#tf.broadcast_to/BroadcastTo/shape/1Const*
_output_shapes
: *
dtype0*
value	B :e
#tf.broadcast_to/BroadcastTo/shape/2Const*
_output_shapes
: *
dtype0*
value	B :╪
!tf.broadcast_to/BroadcastTo/shapePack#tf.compat.v1.floor_div/FloorDiv:z:0,tf.broadcast_to/BroadcastTo/shape/1:output:0,tf.broadcast_to/BroadcastTo/shape/2:output:0*
N*
T0*
_output_shapes
:░
tf.broadcast_to/BroadcastToBroadcastTo"tf.expand_dims/ExpandDims:output:0*tf.broadcast_to/BroadcastTo/shape:output:0*
T0*+
_output_shapes
:         и
tf.math.subtract/SubSub/tf.__operators__.getitem/strided_slice:output:0$tf.broadcast_to/BroadcastTo:output:0*
T0*+
_output_shapes
:         h
&tf.compat.v1.gather_7/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_7/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :°
tf.compat.v1.gather_7/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_7/GatherV2/indices:output:0,tf.compat.v1.gather_7/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         h
&tf.compat.v1.gather_6/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_6/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :°
tf.compat.v1.gather_6/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_6/GatherV2/indices:output:0,tf.compat.v1.gather_6/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         ]
tf.math.multiply_6/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ы
tf.math.multiply_6/MulMul'tf.compat.v1.gather_6/GatherV2:output:0!tf.math.multiply_6/Mul/y:output:0*
T0*'
_output_shapes
:         ]
tf.math.multiply_7/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ы
tf.math.multiply_7/MulMul'tf.compat.v1.gather_7/GatherV2:output:0!tf.math.multiply_7/Mul/y:output:0*
T0*'
_output_shapes
:         [
tf.compat.v1.size_1/SizeSizetf.math.subtract/Sub:z:0*
T0*
_output_shapes
: П
tf.__operators__.add_3/AddV2AddV2tf.math.multiply_6/Mul:z:0tf.math.multiply_7/Mul:z:0*
T0*'
_output_shapes
:         a
tf.expand_dims_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :л
tf.expand_dims_1/ExpandDims
ExpandDims tf.__operators__.add_3/AddV2:z:0(tf.expand_dims_1/ExpandDims/dim:output:0*
T0*+
_output_shapes
:         e
#tf.compat.v1.floor_div_1/FloorDiv/yConst*
_output_shapes
: *
dtype0*
value	B :Я
!tf.compat.v1.floor_div_1/FloorDivFloorDiv!tf.compat.v1.size_1/Size:output:0,tf.compat.v1.floor_div_1/FloorDiv/y:output:0*
T0*
_output_shapes
: h
&tf.compat.v1.gather_3/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_3/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :°
tf.compat.v1.gather_3/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_3/GatherV2/indices:output:0,tf.compat.v1.gather_3/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         h
&tf.compat.v1.gather_2/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_2/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :°
tf.compat.v1.gather_2/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_2/GatherV2/indices:output:0,tf.compat.v1.gather_2/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         h
&tf.compat.v1.gather_5/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_5/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :°
tf.compat.v1.gather_5/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_5/GatherV2/indices:output:0,tf.compat.v1.gather_5/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         h
&tf.compat.v1.gather_4/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_4/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :°
tf.compat.v1.gather_4/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_4/GatherV2/indices:output:0,tf.compat.v1.gather_4/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         g
%tf.broadcast_to_1/BroadcastTo/shape/1Const*
_output_shapes
: *
dtype0*
value	B :g
%tf.broadcast_to_1/BroadcastTo/shape/2Const*
_output_shapes
: *
dtype0*
value	B :р
#tf.broadcast_to_1/BroadcastTo/shapePack%tf.compat.v1.floor_div_1/FloorDiv:z:0.tf.broadcast_to_1/BroadcastTo/shape/1:output:0.tf.broadcast_to_1/BroadcastTo/shape/2:output:0*
N*
T0*
_output_shapes
:╢
tf.broadcast_to_1/BroadcastToBroadcastTo$tf.expand_dims_1/ExpandDims:output:0,tf.broadcast_to_1/BroadcastTo/shape:output:0*
T0*+
_output_shapes
:         ]
tf.math.multiply_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ы
tf.math.multiply_2/MulMul'tf.compat.v1.gather_2/GatherV2:output:0!tf.math.multiply_2/Mul/y:output:0*
T0*'
_output_shapes
:         ]
tf.math.multiply_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ы
tf.math.multiply_3/MulMul'tf.compat.v1.gather_3/GatherV2:output:0!tf.math.multiply_3/Mul/y:output:0*
T0*'
_output_shapes
:         ]
tf.math.multiply_4/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ы
tf.math.multiply_4/MulMul'tf.compat.v1.gather_4/GatherV2:output:0!tf.math.multiply_4/Mul/y:output:0*
T0*'
_output_shapes
:         ]
tf.math.multiply_5/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ы
tf.math.multiply_5/MulMul'tf.compat.v1.gather_5/GatherV2:output:0!tf.math.multiply_5/Mul/y:output:0*
T0*'
_output_shapes
:         Х
tf.math.subtract_2/SubSubtf.math.subtract/Sub:z:0&tf.broadcast_to_1/BroadcastTo:output:0*
T0*+
_output_shapes
:         П
tf.__operators__.add_2/AddV2AddV2tf.math.multiply_4/Mul:z:0tf.math.multiply_5/Mul:z:0*
T0*'
_output_shapes
:         П
tf.__operators__.add_1/AddV2AddV2tf.math.multiply_2/Mul:z:0tf.math.multiply_3/Mul:z:0*
T0*'
_output_shapes
:         h
&tf.compat.v1.gather_8/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B : e
#tf.compat.v1.gather_8/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ё
tf.compat.v1.gather_8/GatherV2GatherV2tf.math.subtract_2/Sub:z:0/tf.compat.v1.gather_8/GatherV2/indices:output:0,tf.compat.v1.gather_8/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes

:У
tf.math.subtract_1/SubSub tf.__operators__.add_2/AddV2:z:0 tf.__operators__.add_1/AddV2:z:0*
T0*'
_output_shapes
:         Ю
tf.compat.v1.norm_1/norm/mulMul'tf.compat.v1.gather_8/GatherV2:output:0'tf.compat.v1.gather_8/GatherV2:output:0*
T0*
_output_shapes

:x
.tf.compat.v1.norm_1/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ╕
tf.compat.v1.norm_1/norm/SumSum tf.compat.v1.norm_1/norm/mul:z:07tf.compat.v1.norm_1/norm/Sum/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(u
tf.compat.v1.norm_1/norm/SqrtSqrt%tf.compat.v1.norm_1/norm/Sum:output:0*
T0*
_output_shapes

:К
 tf.compat.v1.norm_1/norm/SqueezeSqueeze!tf.compat.v1.norm_1/norm/Sqrt:y:0*
T0*
_output_shapes
:*
squeeze_dims
 Л
tf.compat.v1.norm/norm/mulMultf.math.subtract_1/Sub:z:0tf.math.subtract_1/Sub:z:0*
T0*'
_output_shapes
:         m
tf.compat.v1.norm/norm/ConstConst*
_output_shapes
:*
dtype0*
valueB"       в
tf.compat.v1.norm/norm/SumSumtf.compat.v1.norm/norm/mul:z:0%tf.compat.v1.norm/norm/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(q
tf.compat.v1.norm/norm/SqrtSqrt#tf.compat.v1.norm/norm/Sum:output:0*
T0*
_output_shapes

:k
tf.compat.v1.norm/norm/SqueezeSqueezetf.compat.v1.norm/norm/Sqrt:y:0*
T0*
_output_shapes
: ]
tf.math.multiply_8/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @К
tf.math.multiply_8/MulMul'tf.compat.v1.norm/norm/Squeeze:output:0!tf.math.multiply_8/Mul/y:output:0*
T0*
_output_shapes
: b
tf.math.reduce_max/ConstConst*
_output_shapes
:*
dtype0*
valueB: М
tf.math.reduce_max/MaxMax)tf.compat.v1.norm_1/norm/Squeeze:output:0!tf.math.reduce_max/Const:output:0*
T0*
_output_shapes
: А
tf.math.maximum/MaximumMaximumtf.math.multiply_8/Mul:z:0tf.math.reduce_max/Max:output:0*
T0*
_output_shapes
: П
tf.math.truediv/truedivRealDivtf.math.subtract/Sub:z:0tf.math.maximum/Maximum:z:0*
T0*+
_output_shapes
:         ═
flatten_1/PartitionedCallPartitionedCalltf.math.truediv/truediv:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_6670·
dense/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0
dense_6893
dense_6895*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         Z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_6683Ж
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_6898dense_1_6900*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         (*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_6700И
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_6903dense_2_6905*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_6717w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         к
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         ': : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:O K
'
_output_shapes
:         '
 
_user_specified_nameinputs
Щ

Ё
?__inference_dense_layer_call_and_return_conditional_losses_6683

inputs0
matmul_readvariableop_resource:Z-
biasadd_readvariableop_resource:Z
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Z*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Zr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:Z*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ZR
Relu6Relu6BiasAdd:output:0*
T0*'
_output_shapes
:         Zb
IdentityIdentityRelu6:activations:0^NoOp*
T0*'
_output_shapes
:         Zw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╒^
С
 __inference__traced_restore_7716
file_prefix/
assignvariableop_dense_kernel:Z+
assignvariableop_1_dense_bias:Z3
!assignvariableop_2_dense_1_kernel:Z(-
assignvariableop_3_dense_1_bias:(3
!assignvariableop_4_dense_2_kernel:(-
assignvariableop_5_dense_2_bias:&
assignvariableop_6_iteration:	 *
 assignvariableop_7_learning_rate: 8
&assignvariableop_8_adam_m_dense_kernel:Z8
&assignvariableop_9_adam_v_dense_kernel:Z3
%assignvariableop_10_adam_m_dense_bias:Z3
%assignvariableop_11_adam_v_dense_bias:Z;
)assignvariableop_12_adam_m_dense_1_kernel:Z(;
)assignvariableop_13_adam_v_dense_1_kernel:Z(5
'assignvariableop_14_adam_m_dense_1_bias:(5
'assignvariableop_15_adam_v_dense_1_bias:(;
)assignvariableop_16_adam_m_dense_2_kernel:(;
)assignvariableop_17_adam_v_dense_2_kernel:(5
'assignvariableop_18_adam_m_dense_2_bias:5
'assignvariableop_19_adam_v_dense_2_bias:#
assignvariableop_20_total: #
assignvariableop_21_count: 
identity_23ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9С

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*╖	
valueн	Bк	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЮ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B С
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*p
_output_shapes^
\:::::::::::::::::::::::*%
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:░
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:╕
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:╢
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:╕
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_2_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:╢
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_2_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:│
AssignVariableOp_6AssignVariableOpassignvariableop_6_iterationIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:╖
AssignVariableOp_7AssignVariableOp assignvariableop_7_learning_rateIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:╜
AssignVariableOp_8AssignVariableOp&assignvariableop_8_adam_m_dense_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:╜
AssignVariableOp_9AssignVariableOp&assignvariableop_9_adam_v_dense_kernelIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_10AssignVariableOp%assignvariableop_10_adam_m_dense_biasIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_11AssignVariableOp%assignvariableop_11_adam_v_dense_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_12AssignVariableOp)assignvariableop_12_adam_m_dense_1_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_13AssignVariableOp)assignvariableop_13_adam_v_dense_1_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_14AssignVariableOp'assignvariableop_14_adam_m_dense_1_biasIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_15AssignVariableOp'assignvariableop_15_adam_v_dense_1_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_16AssignVariableOp)assignvariableop_16_adam_m_dense_2_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_17AssignVariableOp)assignvariableop_17_adam_v_dense_2_kernelIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_18AssignVariableOp'assignvariableop_18_adam_m_dense_2_biasIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_19AssignVariableOp'assignvariableop_19_adam_v_dense_2_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_20AssignVariableOpassignvariableop_20_totalIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_21AssignVariableOpassignvariableop_21_countIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 │
Identity_22Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_23IdentityIdentity_22:output:0^NoOp_1*
T0*
_output_shapes
: а
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_23Identity_23:output:0*A
_input_shapes0
.: : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ч

Є
A__inference_dense_2_layer_call_and_return_conditional_losses_7551

inputs0
matmul_readvariableop_resource:(-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         (: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         (
 
_user_specified_nameinputs
г
D
(__inference_reshape_1_layer_call_fn_7467

inputs
identity▓
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_reshape_1_layer_call_and_return_conditional_losses_6574d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         ':O K
'
_output_shapes
:         '
 
_user_specified_nameinputs
їИ
ь
?__inference_model_layer_call_and_return_conditional_losses_7462

inputs6
$dense_matmul_readvariableop_resource:Z3
%dense_biasadd_readvariableop_resource:Z8
&dense_1_matmul_readvariableop_resource:Z(5
'dense_1_biasadd_readvariableop_resource:(8
&dense_2_matmul_readvariableop_resource:(5
'dense_2_biasadd_readvariableop_resource:
identityИвdense/BiasAdd/ReadVariableOpвdense/MatMul/ReadVariableOpвdense_1/BiasAdd/ReadVariableOpвdense_1/MatMul/ReadVariableOpвdense_2/BiasAdd/ReadVariableOpвdense_2/MatMul/ReadVariableOpE
reshape_1/ShapeShapeinputs*
T0*
_output_shapes
:g
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :╖
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:|
reshape_1/ReshapeReshapeinputs reshape_1/Reshape/shape:output:0*
T0*+
_output_shapes
:         Б
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            Г
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           Г
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         р
&tf.__operators__.getitem/strided_sliceStridedSlicereshape_1/Reshape:output:05tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         *

begin_mask*
end_maskh
&tf.compat.v1.gather_1/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :П
tf.compat.v1.gather_1/GatherV2GatherV2/tf.__operators__.getitem/strided_slice:output:0/tf.compat.v1.gather_1/GatherV2/indices:output:0,tf.compat.v1.gather_1/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         f
$tf.compat.v1.gather/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :c
!tf.compat.v1.gather/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :Й
tf.compat.v1.gather/GatherV2GatherV2/tf.__operators__.getitem/strided_slice:output:0-tf.compat.v1.gather/GatherV2/indices:output:0*tf.compat.v1.gather/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         [
tf.math.multiply/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Х
tf.math.multiply/MulMul%tf.compat.v1.gather/GatherV2:output:0tf.math.multiply/Mul/y:output:0*
T0*'
_output_shapes
:         ]
tf.math.multiply_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ы
tf.math.multiply_1/MulMul'tf.compat.v1.gather_1/GatherV2:output:0!tf.math.multiply_1/Mul/y:output:0*
T0*'
_output_shapes
:         p
tf.compat.v1.size/SizeSize/tf.__operators__.getitem/strided_slice:output:0*
T0*
_output_shapes
: Л
tf.__operators__.add/AddV2AddV2tf.math.multiply/Mul:z:0tf.math.multiply_1/Mul:z:0*
T0*'
_output_shapes
:         _
tf.expand_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :е
tf.expand_dims/ExpandDims
ExpandDimstf.__operators__.add/AddV2:z:0&tf.expand_dims/ExpandDims/dim:output:0*
T0*+
_output_shapes
:         c
!tf.compat.v1.floor_div/FloorDiv/yConst*
_output_shapes
: *
dtype0*
value	B :Щ
tf.compat.v1.floor_div/FloorDivFloorDivtf.compat.v1.size/Size:output:0*tf.compat.v1.floor_div/FloorDiv/y:output:0*
T0*
_output_shapes
: e
#tf.broadcast_to/BroadcastTo/shape/1Const*
_output_shapes
: *
dtype0*
value	B :e
#tf.broadcast_to/BroadcastTo/shape/2Const*
_output_shapes
: *
dtype0*
value	B :╪
!tf.broadcast_to/BroadcastTo/shapePack#tf.compat.v1.floor_div/FloorDiv:z:0,tf.broadcast_to/BroadcastTo/shape/1:output:0,tf.broadcast_to/BroadcastTo/shape/2:output:0*
N*
T0*
_output_shapes
:░
tf.broadcast_to/BroadcastToBroadcastTo"tf.expand_dims/ExpandDims:output:0*tf.broadcast_to/BroadcastTo/shape:output:0*
T0*+
_output_shapes
:         и
tf.math.subtract/SubSub/tf.__operators__.getitem/strided_slice:output:0$tf.broadcast_to/BroadcastTo:output:0*
T0*+
_output_shapes
:         h
&tf.compat.v1.gather_7/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_7/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :°
tf.compat.v1.gather_7/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_7/GatherV2/indices:output:0,tf.compat.v1.gather_7/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         h
&tf.compat.v1.gather_6/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_6/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :°
tf.compat.v1.gather_6/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_6/GatherV2/indices:output:0,tf.compat.v1.gather_6/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         ]
tf.math.multiply_6/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ы
tf.math.multiply_6/MulMul'tf.compat.v1.gather_6/GatherV2:output:0!tf.math.multiply_6/Mul/y:output:0*
T0*'
_output_shapes
:         ]
tf.math.multiply_7/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ы
tf.math.multiply_7/MulMul'tf.compat.v1.gather_7/GatherV2:output:0!tf.math.multiply_7/Mul/y:output:0*
T0*'
_output_shapes
:         [
tf.compat.v1.size_1/SizeSizetf.math.subtract/Sub:z:0*
T0*
_output_shapes
: П
tf.__operators__.add_3/AddV2AddV2tf.math.multiply_6/Mul:z:0tf.math.multiply_7/Mul:z:0*
T0*'
_output_shapes
:         a
tf.expand_dims_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :л
tf.expand_dims_1/ExpandDims
ExpandDims tf.__operators__.add_3/AddV2:z:0(tf.expand_dims_1/ExpandDims/dim:output:0*
T0*+
_output_shapes
:         e
#tf.compat.v1.floor_div_1/FloorDiv/yConst*
_output_shapes
: *
dtype0*
value	B :Я
!tf.compat.v1.floor_div_1/FloorDivFloorDiv!tf.compat.v1.size_1/Size:output:0,tf.compat.v1.floor_div_1/FloorDiv/y:output:0*
T0*
_output_shapes
: h
&tf.compat.v1.gather_3/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_3/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :°
tf.compat.v1.gather_3/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_3/GatherV2/indices:output:0,tf.compat.v1.gather_3/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         h
&tf.compat.v1.gather_2/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_2/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :°
tf.compat.v1.gather_2/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_2/GatherV2/indices:output:0,tf.compat.v1.gather_2/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         h
&tf.compat.v1.gather_5/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_5/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :°
tf.compat.v1.gather_5/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_5/GatherV2/indices:output:0,tf.compat.v1.gather_5/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         h
&tf.compat.v1.gather_4/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_4/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :°
tf.compat.v1.gather_4/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_4/GatherV2/indices:output:0,tf.compat.v1.gather_4/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         g
%tf.broadcast_to_1/BroadcastTo/shape/1Const*
_output_shapes
: *
dtype0*
value	B :g
%tf.broadcast_to_1/BroadcastTo/shape/2Const*
_output_shapes
: *
dtype0*
value	B :р
#tf.broadcast_to_1/BroadcastTo/shapePack%tf.compat.v1.floor_div_1/FloorDiv:z:0.tf.broadcast_to_1/BroadcastTo/shape/1:output:0.tf.broadcast_to_1/BroadcastTo/shape/2:output:0*
N*
T0*
_output_shapes
:╢
tf.broadcast_to_1/BroadcastToBroadcastTo$tf.expand_dims_1/ExpandDims:output:0,tf.broadcast_to_1/BroadcastTo/shape:output:0*
T0*+
_output_shapes
:         ]
tf.math.multiply_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ы
tf.math.multiply_2/MulMul'tf.compat.v1.gather_2/GatherV2:output:0!tf.math.multiply_2/Mul/y:output:0*
T0*'
_output_shapes
:         ]
tf.math.multiply_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ы
tf.math.multiply_3/MulMul'tf.compat.v1.gather_3/GatherV2:output:0!tf.math.multiply_3/Mul/y:output:0*
T0*'
_output_shapes
:         ]
tf.math.multiply_4/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ы
tf.math.multiply_4/MulMul'tf.compat.v1.gather_4/GatherV2:output:0!tf.math.multiply_4/Mul/y:output:0*
T0*'
_output_shapes
:         ]
tf.math.multiply_5/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ы
tf.math.multiply_5/MulMul'tf.compat.v1.gather_5/GatherV2:output:0!tf.math.multiply_5/Mul/y:output:0*
T0*'
_output_shapes
:         Х
tf.math.subtract_2/SubSubtf.math.subtract/Sub:z:0&tf.broadcast_to_1/BroadcastTo:output:0*
T0*+
_output_shapes
:         П
tf.__operators__.add_2/AddV2AddV2tf.math.multiply_4/Mul:z:0tf.math.multiply_5/Mul:z:0*
T0*'
_output_shapes
:         П
tf.__operators__.add_1/AddV2AddV2tf.math.multiply_2/Mul:z:0tf.math.multiply_3/Mul:z:0*
T0*'
_output_shapes
:         h
&tf.compat.v1.gather_8/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B : e
#tf.compat.v1.gather_8/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ё
tf.compat.v1.gather_8/GatherV2GatherV2tf.math.subtract_2/Sub:z:0/tf.compat.v1.gather_8/GatherV2/indices:output:0,tf.compat.v1.gather_8/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes

:У
tf.math.subtract_1/SubSub tf.__operators__.add_2/AddV2:z:0 tf.__operators__.add_1/AddV2:z:0*
T0*'
_output_shapes
:         Ю
tf.compat.v1.norm_1/norm/mulMul'tf.compat.v1.gather_8/GatherV2:output:0'tf.compat.v1.gather_8/GatherV2:output:0*
T0*
_output_shapes

:x
.tf.compat.v1.norm_1/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ╕
tf.compat.v1.norm_1/norm/SumSum tf.compat.v1.norm_1/norm/mul:z:07tf.compat.v1.norm_1/norm/Sum/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(u
tf.compat.v1.norm_1/norm/SqrtSqrt%tf.compat.v1.norm_1/norm/Sum:output:0*
T0*
_output_shapes

:К
 tf.compat.v1.norm_1/norm/SqueezeSqueeze!tf.compat.v1.norm_1/norm/Sqrt:y:0*
T0*
_output_shapes
:*
squeeze_dims
 Л
tf.compat.v1.norm/norm/mulMultf.math.subtract_1/Sub:z:0tf.math.subtract_1/Sub:z:0*
T0*'
_output_shapes
:         m
tf.compat.v1.norm/norm/ConstConst*
_output_shapes
:*
dtype0*
valueB"       в
tf.compat.v1.norm/norm/SumSumtf.compat.v1.norm/norm/mul:z:0%tf.compat.v1.norm/norm/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(q
tf.compat.v1.norm/norm/SqrtSqrt#tf.compat.v1.norm/norm/Sum:output:0*
T0*
_output_shapes

:k
tf.compat.v1.norm/norm/SqueezeSqueezetf.compat.v1.norm/norm/Sqrt:y:0*
T0*
_output_shapes
: ]
tf.math.multiply_8/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @К
tf.math.multiply_8/MulMul'tf.compat.v1.norm/norm/Squeeze:output:0!tf.math.multiply_8/Mul/y:output:0*
T0*
_output_shapes
: b
tf.math.reduce_max/ConstConst*
_output_shapes
:*
dtype0*
valueB: М
tf.math.reduce_max/MaxMax)tf.compat.v1.norm_1/norm/Squeeze:output:0!tf.math.reduce_max/Const:output:0*
T0*
_output_shapes
: А
tf.math.maximum/MaximumMaximumtf.math.multiply_8/Mul:z:0tf.math.reduce_max/Max:output:0*
T0*
_output_shapes
: П
tf.math.truediv/truedivRealDivtf.math.subtract/Sub:z:0tf.math.maximum/Maximum:z:0*
T0*+
_output_shapes
:         `
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Е
flatten_1/ReshapeReshapetf.math.truediv/truediv:z:0flatten_1/Const:output:0*
T0*'
_output_shapes
:         А
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:Z*
dtype0Й
dense/MatMulMatMulflatten_1/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Z~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:Z*
dtype0И
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Z^
dense/Relu6Relu6dense/BiasAdd:output:0*
T0*'
_output_shapes
:         ZД
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:Z(*
dtype0М
dense_1/MatMulMatMuldense/Relu6:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         (В
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0О
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         (b
dense_1/Relu6Relu6dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         (Д
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0О
dense_2/MatMulMatMuldense_1/Relu6:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         В
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         f
dense_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*'
_output_shapes
:         b
IdentityIdentitydense_2/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:         Е
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         ': : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:O K
'
_output_shapes
:         '
 
_user_specified_nameinputs
┌Х
Х
__inference__wrapped_model_6554
input_1<
*model_dense_matmul_readvariableop_resource:Z9
+model_dense_biasadd_readvariableop_resource:Z>
,model_dense_1_matmul_readvariableop_resource:Z(;
-model_dense_1_biasadd_readvariableop_resource:(>
,model_dense_2_matmul_readvariableop_resource:(;
-model_dense_2_biasadd_readvariableop_resource:
identityИв"model/dense/BiasAdd/ReadVariableOpв!model/dense/MatMul/ReadVariableOpв$model/dense_1/BiasAdd/ReadVariableOpв#model/dense_1/MatMul/ReadVariableOpв$model/dense_2/BiasAdd/ReadVariableOpв#model/dense_2/MatMul/ReadVariableOpL
model/reshape_1/ShapeShapeinput_1*
T0*
_output_shapes
:m
#model/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%model/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%model/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
model/reshape_1/strided_sliceStridedSlicemodel/reshape_1/Shape:output:0,model/reshape_1/strided_slice/stack:output:0.model/reshape_1/strided_slice/stack_1:output:0.model/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
model/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :a
model/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :╧
model/reshape_1/Reshape/shapePack&model/reshape_1/strided_slice:output:0(model/reshape_1/Reshape/shape/1:output:0(model/reshape_1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:Й
model/reshape_1/ReshapeReshapeinput_1&model/reshape_1/Reshape/shape:output:0*
T0*+
_output_shapes
:         З
2model/tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            Й
4model/tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           Й
4model/tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ■
,model/tf.__operators__.getitem/strided_sliceStridedSlice model/reshape_1/Reshape:output:0;model/tf.__operators__.getitem/strided_slice/stack:output:0=model/tf.__operators__.getitem/strided_slice/stack_1:output:0=model/tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         *

begin_mask*
end_maskn
,model/tf.compat.v1.gather_1/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :k
)model/tf.compat.v1.gather_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :з
$model/tf.compat.v1.gather_1/GatherV2GatherV25model/tf.__operators__.getitem/strided_slice:output:05model/tf.compat.v1.gather_1/GatherV2/indices:output:02model/tf.compat.v1.gather_1/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         l
*model/tf.compat.v1.gather/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :i
'model/tf.compat.v1.gather/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :б
"model/tf.compat.v1.gather/GatherV2GatherV25model/tf.__operators__.getitem/strided_slice:output:03model/tf.compat.v1.gather/GatherV2/indices:output:00model/tf.compat.v1.gather/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         a
model/tf.math.multiply/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?з
model/tf.math.multiply/MulMul+model/tf.compat.v1.gather/GatherV2:output:0%model/tf.math.multiply/Mul/y:output:0*
T0*'
_output_shapes
:         c
model/tf.math.multiply_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?н
model/tf.math.multiply_1/MulMul-model/tf.compat.v1.gather_1/GatherV2:output:0'model/tf.math.multiply_1/Mul/y:output:0*
T0*'
_output_shapes
:         |
model/tf.compat.v1.size/SizeSize5model/tf.__operators__.getitem/strided_slice:output:0*
T0*
_output_shapes
: Э
 model/tf.__operators__.add/AddV2AddV2model/tf.math.multiply/Mul:z:0 model/tf.math.multiply_1/Mul:z:0*
T0*'
_output_shapes
:         e
#model/tf.expand_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :╖
model/tf.expand_dims/ExpandDims
ExpandDims$model/tf.__operators__.add/AddV2:z:0,model/tf.expand_dims/ExpandDims/dim:output:0*
T0*+
_output_shapes
:         i
'model/tf.compat.v1.floor_div/FloorDiv/yConst*
_output_shapes
: *
dtype0*
value	B :л
%model/tf.compat.v1.floor_div/FloorDivFloorDiv%model/tf.compat.v1.size/Size:output:00model/tf.compat.v1.floor_div/FloorDiv/y:output:0*
T0*
_output_shapes
: k
)model/tf.broadcast_to/BroadcastTo/shape/1Const*
_output_shapes
: *
dtype0*
value	B :k
)model/tf.broadcast_to/BroadcastTo/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Ё
'model/tf.broadcast_to/BroadcastTo/shapePack)model/tf.compat.v1.floor_div/FloorDiv:z:02model/tf.broadcast_to/BroadcastTo/shape/1:output:02model/tf.broadcast_to/BroadcastTo/shape/2:output:0*
N*
T0*
_output_shapes
:┬
!model/tf.broadcast_to/BroadcastToBroadcastTo(model/tf.expand_dims/ExpandDims:output:00model/tf.broadcast_to/BroadcastTo/shape:output:0*
T0*+
_output_shapes
:         ║
model/tf.math.subtract/SubSub5model/tf.__operators__.getitem/strided_slice:output:0*model/tf.broadcast_to/BroadcastTo:output:0*
T0*+
_output_shapes
:         n
,model/tf.compat.v1.gather_7/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :k
)model/tf.compat.v1.gather_7/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :Р
$model/tf.compat.v1.gather_7/GatherV2GatherV2model/tf.math.subtract/Sub:z:05model/tf.compat.v1.gather_7/GatherV2/indices:output:02model/tf.compat.v1.gather_7/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         n
,model/tf.compat.v1.gather_6/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :k
)model/tf.compat.v1.gather_6/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :Р
$model/tf.compat.v1.gather_6/GatherV2GatherV2model/tf.math.subtract/Sub:z:05model/tf.compat.v1.gather_6/GatherV2/indices:output:02model/tf.compat.v1.gather_6/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         c
model/tf.math.multiply_6/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?н
model/tf.math.multiply_6/MulMul-model/tf.compat.v1.gather_6/GatherV2:output:0'model/tf.math.multiply_6/Mul/y:output:0*
T0*'
_output_shapes
:         c
model/tf.math.multiply_7/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?н
model/tf.math.multiply_7/MulMul-model/tf.compat.v1.gather_7/GatherV2:output:0'model/tf.math.multiply_7/Mul/y:output:0*
T0*'
_output_shapes
:         g
model/tf.compat.v1.size_1/SizeSizemodel/tf.math.subtract/Sub:z:0*
T0*
_output_shapes
: б
"model/tf.__operators__.add_3/AddV2AddV2 model/tf.math.multiply_6/Mul:z:0 model/tf.math.multiply_7/Mul:z:0*
T0*'
_output_shapes
:         g
%model/tf.expand_dims_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :╜
!model/tf.expand_dims_1/ExpandDims
ExpandDims&model/tf.__operators__.add_3/AddV2:z:0.model/tf.expand_dims_1/ExpandDims/dim:output:0*
T0*+
_output_shapes
:         k
)model/tf.compat.v1.floor_div_1/FloorDiv/yConst*
_output_shapes
: *
dtype0*
value	B :▒
'model/tf.compat.v1.floor_div_1/FloorDivFloorDiv'model/tf.compat.v1.size_1/Size:output:02model/tf.compat.v1.floor_div_1/FloorDiv/y:output:0*
T0*
_output_shapes
: n
,model/tf.compat.v1.gather_3/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :k
)model/tf.compat.v1.gather_3/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :Р
$model/tf.compat.v1.gather_3/GatherV2GatherV2model/tf.math.subtract/Sub:z:05model/tf.compat.v1.gather_3/GatherV2/indices:output:02model/tf.compat.v1.gather_3/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         n
,model/tf.compat.v1.gather_2/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :k
)model/tf.compat.v1.gather_2/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :Р
$model/tf.compat.v1.gather_2/GatherV2GatherV2model/tf.math.subtract/Sub:z:05model/tf.compat.v1.gather_2/GatherV2/indices:output:02model/tf.compat.v1.gather_2/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         n
,model/tf.compat.v1.gather_5/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :k
)model/tf.compat.v1.gather_5/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :Р
$model/tf.compat.v1.gather_5/GatherV2GatherV2model/tf.math.subtract/Sub:z:05model/tf.compat.v1.gather_5/GatherV2/indices:output:02model/tf.compat.v1.gather_5/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         n
,model/tf.compat.v1.gather_4/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :k
)model/tf.compat.v1.gather_4/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :Р
$model/tf.compat.v1.gather_4/GatherV2GatherV2model/tf.math.subtract/Sub:z:05model/tf.compat.v1.gather_4/GatherV2/indices:output:02model/tf.compat.v1.gather_4/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         m
+model/tf.broadcast_to_1/BroadcastTo/shape/1Const*
_output_shapes
: *
dtype0*
value	B :m
+model/tf.broadcast_to_1/BroadcastTo/shape/2Const*
_output_shapes
: *
dtype0*
value	B :°
)model/tf.broadcast_to_1/BroadcastTo/shapePack+model/tf.compat.v1.floor_div_1/FloorDiv:z:04model/tf.broadcast_to_1/BroadcastTo/shape/1:output:04model/tf.broadcast_to_1/BroadcastTo/shape/2:output:0*
N*
T0*
_output_shapes
:╚
#model/tf.broadcast_to_1/BroadcastToBroadcastTo*model/tf.expand_dims_1/ExpandDims:output:02model/tf.broadcast_to_1/BroadcastTo/shape:output:0*
T0*+
_output_shapes
:         c
model/tf.math.multiply_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?н
model/tf.math.multiply_2/MulMul-model/tf.compat.v1.gather_2/GatherV2:output:0'model/tf.math.multiply_2/Mul/y:output:0*
T0*'
_output_shapes
:         c
model/tf.math.multiply_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?н
model/tf.math.multiply_3/MulMul-model/tf.compat.v1.gather_3/GatherV2:output:0'model/tf.math.multiply_3/Mul/y:output:0*
T0*'
_output_shapes
:         c
model/tf.math.multiply_4/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?н
model/tf.math.multiply_4/MulMul-model/tf.compat.v1.gather_4/GatherV2:output:0'model/tf.math.multiply_4/Mul/y:output:0*
T0*'
_output_shapes
:         c
model/tf.math.multiply_5/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?н
model/tf.math.multiply_5/MulMul-model/tf.compat.v1.gather_5/GatherV2:output:0'model/tf.math.multiply_5/Mul/y:output:0*
T0*'
_output_shapes
:         з
model/tf.math.subtract_2/SubSubmodel/tf.math.subtract/Sub:z:0,model/tf.broadcast_to_1/BroadcastTo:output:0*
T0*+
_output_shapes
:         б
"model/tf.__operators__.add_2/AddV2AddV2 model/tf.math.multiply_4/Mul:z:0 model/tf.math.multiply_5/Mul:z:0*
T0*'
_output_shapes
:         б
"model/tf.__operators__.add_1/AddV2AddV2 model/tf.math.multiply_2/Mul:z:0 model/tf.math.multiply_3/Mul:z:0*
T0*'
_output_shapes
:         n
,model/tf.compat.v1.gather_8/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B : k
)model/tf.compat.v1.gather_8/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Й
$model/tf.compat.v1.gather_8/GatherV2GatherV2 model/tf.math.subtract_2/Sub:z:05model/tf.compat.v1.gather_8/GatherV2/indices:output:02model/tf.compat.v1.gather_8/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes

:е
model/tf.math.subtract_1/SubSub&model/tf.__operators__.add_2/AddV2:z:0&model/tf.__operators__.add_1/AddV2:z:0*
T0*'
_output_shapes
:         ░
"model/tf.compat.v1.norm_1/norm/mulMul-model/tf.compat.v1.gather_8/GatherV2:output:0-model/tf.compat.v1.gather_8/GatherV2:output:0*
T0*
_output_shapes

:~
4model/tf.compat.v1.norm_1/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ╩
"model/tf.compat.v1.norm_1/norm/SumSum&model/tf.compat.v1.norm_1/norm/mul:z:0=model/tf.compat.v1.norm_1/norm/Sum/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(Б
#model/tf.compat.v1.norm_1/norm/SqrtSqrt+model/tf.compat.v1.norm_1/norm/Sum:output:0*
T0*
_output_shapes

:Ц
&model/tf.compat.v1.norm_1/norm/SqueezeSqueeze'model/tf.compat.v1.norm_1/norm/Sqrt:y:0*
T0*
_output_shapes
:*
squeeze_dims
 Э
 model/tf.compat.v1.norm/norm/mulMul model/tf.math.subtract_1/Sub:z:0 model/tf.math.subtract_1/Sub:z:0*
T0*'
_output_shapes
:         s
"model/tf.compat.v1.norm/norm/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ┤
 model/tf.compat.v1.norm/norm/SumSum$model/tf.compat.v1.norm/norm/mul:z:0+model/tf.compat.v1.norm/norm/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(}
!model/tf.compat.v1.norm/norm/SqrtSqrt)model/tf.compat.v1.norm/norm/Sum:output:0*
T0*
_output_shapes

:w
$model/tf.compat.v1.norm/norm/SqueezeSqueeze%model/tf.compat.v1.norm/norm/Sqrt:y:0*
T0*
_output_shapes
: c
model/tf.math.multiply_8/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @Ь
model/tf.math.multiply_8/MulMul-model/tf.compat.v1.norm/norm/Squeeze:output:0'model/tf.math.multiply_8/Mul/y:output:0*
T0*
_output_shapes
: h
model/tf.math.reduce_max/ConstConst*
_output_shapes
:*
dtype0*
valueB: Ю
model/tf.math.reduce_max/MaxMax/model/tf.compat.v1.norm_1/norm/Squeeze:output:0'model/tf.math.reduce_max/Const:output:0*
T0*
_output_shapes
: Т
model/tf.math.maximum/MaximumMaximum model/tf.math.multiply_8/Mul:z:0%model/tf.math.reduce_max/Max:output:0*
T0*
_output_shapes
: б
model/tf.math.truediv/truedivRealDivmodel/tf.math.subtract/Sub:z:0!model/tf.math.maximum/Maximum:z:0*
T0*+
_output_shapes
:         f
model/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Ч
model/flatten_1/ReshapeReshape!model/tf.math.truediv/truediv:z:0model/flatten_1/Const:output:0*
T0*'
_output_shapes
:         М
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes

:Z*
dtype0Ы
model/dense/MatMulMatMul model/flatten_1/Reshape:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ZК
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:Z*
dtype0Ъ
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Zj
model/dense/Relu6Relu6model/dense/BiasAdd:output:0*
T0*'
_output_shapes
:         ZР
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes

:Z(*
dtype0Ю
model/dense_1/MatMulMatMulmodel/dense/Relu6:activations:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         (О
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0а
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         (n
model/dense_1/Relu6Relu6model/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         (Р
#model/dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0а
model/dense_2/MatMulMatMul!model/dense_1/Relu6:activations:0+model/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         О
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0а
model/dense_2/BiasAddBiasAddmodel/dense_2/MatMul:product:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
model/dense_2/SigmoidSigmoidmodel/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:         h
IdentityIdentitymodel/dense_2/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:         й
NoOpNoOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp$^model/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         ': : : : : : 2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2L
$model/dense_2/BiasAdd/ReadVariableOp$model/dense_2/BiasAdd/ReadVariableOp2J
#model/dense_2/MatMul/ReadVariableOp#model/dense_2/MatMul/ReadVariableOp:P L
'
_output_shapes
:         '
!
_user_specified_name	input_1
р
¤
$__inference_model_layer_call_fn_7214

inputs
unknown:Z
	unknown_0:Z
	unknown_1:Z(
	unknown_2:(
	unknown_3:(
	unknown_4:
identityИвStatefulPartitionedCallИ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_6909o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         ': : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         '
 
_user_specified_nameinputs
Ы

Є
A__inference_dense_1_layer_call_and_return_conditional_losses_7531

inputs0
matmul_readvariableop_resource:Z(-
biasadd_readvariableop_resource:(
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Z(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         (r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         (R
Relu6Relu6BiasAdd:output:0*
T0*'
_output_shapes
:         (b
IdentityIdentityRelu6:activations:0^NoOp*
T0*'
_output_shapes
:         (w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         Z: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         Z
 
_user_specified_nameinputs
Щ

Ё
?__inference_dense_layer_call_and_return_conditional_losses_7511

inputs0
matmul_readvariableop_resource:Z-
biasadd_readvariableop_resource:Z
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Z*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Zr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:Z*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ZR
Relu6Relu6BiasAdd:output:0*
T0*'
_output_shapes
:         Zb
IdentityIdentityRelu6:activations:0^NoOp*
T0*'
_output_shapes
:         Zw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
г
D
(__inference_flatten_1_layer_call_fn_7485

inputs
identityо
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_6670`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         :S O
+
_output_shapes
:         
 
_user_specified_nameinputs
╒

_
C__inference_reshape_1_layer_call_and_return_conditional_losses_6574

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :П
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:h
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:         \
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         ':O K
'
_output_shapes
:         '
 
_user_specified_nameinputs
╝
У
&__inference_dense_1_layer_call_fn_7520

inputs
unknown:Z(
	unknown_0:(
identityИвStatefulPartitionedCall╓
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         (*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_6700o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         (`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         Z: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         Z
 
_user_specified_nameinputs
┴
№
"__inference_signature_wrapper_7180
input_1
unknown:Z
	unknown_0:Z
	unknown_1:Z(
	unknown_2:(
	unknown_3:(
	unknown_4:
identityИвStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *(
f#R!
__inference__wrapped_model_6554o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         ': : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:         '
!
_user_specified_name	input_1
╗
_
C__inference_flatten_1_layer_call_and_return_conditional_losses_7491

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:         X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         :S O
+
_output_shapes
:         
 
_user_specified_nameinputs
╒

_
C__inference_reshape_1_layer_call_and_return_conditional_losses_7480

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :П
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:h
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:         \
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         ':O K
'
_output_shapes
:         '
 
_user_specified_nameinputs
вy
є
?__inference_model_layer_call_and_return_conditional_losses_7159
input_1

dense_7143:Z

dense_7145:Z
dense_1_7148:Z(
dense_1_7150:(
dense_2_7153:(
dense_2_7155:
identityИвdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCallвdense_2/StatefulPartitionedCall╜
reshape_1/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_reshape_1_layer_call_and_return_conditional_losses_6574Б
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            Г
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           Г
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ш
&tf.__operators__.getitem/strided_sliceStridedSlice"reshape_1/PartitionedCall:output:05tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         *

begin_mask*
end_maskh
&tf.compat.v1.gather_1/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :П
tf.compat.v1.gather_1/GatherV2GatherV2/tf.__operators__.getitem/strided_slice:output:0/tf.compat.v1.gather_1/GatherV2/indices:output:0,tf.compat.v1.gather_1/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         f
$tf.compat.v1.gather/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :c
!tf.compat.v1.gather/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :Й
tf.compat.v1.gather/GatherV2GatherV2/tf.__operators__.getitem/strided_slice:output:0-tf.compat.v1.gather/GatherV2/indices:output:0*tf.compat.v1.gather/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         [
tf.math.multiply/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Х
tf.math.multiply/MulMul%tf.compat.v1.gather/GatherV2:output:0tf.math.multiply/Mul/y:output:0*
T0*'
_output_shapes
:         ]
tf.math.multiply_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ы
tf.math.multiply_1/MulMul'tf.compat.v1.gather_1/GatherV2:output:0!tf.math.multiply_1/Mul/y:output:0*
T0*'
_output_shapes
:         p
tf.compat.v1.size/SizeSize/tf.__operators__.getitem/strided_slice:output:0*
T0*
_output_shapes
: Л
tf.__operators__.add/AddV2AddV2tf.math.multiply/Mul:z:0tf.math.multiply_1/Mul:z:0*
T0*'
_output_shapes
:         _
tf.expand_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :е
tf.expand_dims/ExpandDims
ExpandDimstf.__operators__.add/AddV2:z:0&tf.expand_dims/ExpandDims/dim:output:0*
T0*+
_output_shapes
:         c
!tf.compat.v1.floor_div/FloorDiv/yConst*
_output_shapes
: *
dtype0*
value	B :Щ
tf.compat.v1.floor_div/FloorDivFloorDivtf.compat.v1.size/Size:output:0*tf.compat.v1.floor_div/FloorDiv/y:output:0*
T0*
_output_shapes
: e
#tf.broadcast_to/BroadcastTo/shape/1Const*
_output_shapes
: *
dtype0*
value	B :e
#tf.broadcast_to/BroadcastTo/shape/2Const*
_output_shapes
: *
dtype0*
value	B :╪
!tf.broadcast_to/BroadcastTo/shapePack#tf.compat.v1.floor_div/FloorDiv:z:0,tf.broadcast_to/BroadcastTo/shape/1:output:0,tf.broadcast_to/BroadcastTo/shape/2:output:0*
N*
T0*
_output_shapes
:░
tf.broadcast_to/BroadcastToBroadcastTo"tf.expand_dims/ExpandDims:output:0*tf.broadcast_to/BroadcastTo/shape:output:0*
T0*+
_output_shapes
:         и
tf.math.subtract/SubSub/tf.__operators__.getitem/strided_slice:output:0$tf.broadcast_to/BroadcastTo:output:0*
T0*+
_output_shapes
:         h
&tf.compat.v1.gather_7/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_7/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :°
tf.compat.v1.gather_7/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_7/GatherV2/indices:output:0,tf.compat.v1.gather_7/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         h
&tf.compat.v1.gather_6/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_6/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :°
tf.compat.v1.gather_6/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_6/GatherV2/indices:output:0,tf.compat.v1.gather_6/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         ]
tf.math.multiply_6/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ы
tf.math.multiply_6/MulMul'tf.compat.v1.gather_6/GatherV2:output:0!tf.math.multiply_6/Mul/y:output:0*
T0*'
_output_shapes
:         ]
tf.math.multiply_7/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ы
tf.math.multiply_7/MulMul'tf.compat.v1.gather_7/GatherV2:output:0!tf.math.multiply_7/Mul/y:output:0*
T0*'
_output_shapes
:         [
tf.compat.v1.size_1/SizeSizetf.math.subtract/Sub:z:0*
T0*
_output_shapes
: П
tf.__operators__.add_3/AddV2AddV2tf.math.multiply_6/Mul:z:0tf.math.multiply_7/Mul:z:0*
T0*'
_output_shapes
:         a
tf.expand_dims_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :л
tf.expand_dims_1/ExpandDims
ExpandDims tf.__operators__.add_3/AddV2:z:0(tf.expand_dims_1/ExpandDims/dim:output:0*
T0*+
_output_shapes
:         e
#tf.compat.v1.floor_div_1/FloorDiv/yConst*
_output_shapes
: *
dtype0*
value	B :Я
!tf.compat.v1.floor_div_1/FloorDivFloorDiv!tf.compat.v1.size_1/Size:output:0,tf.compat.v1.floor_div_1/FloorDiv/y:output:0*
T0*
_output_shapes
: h
&tf.compat.v1.gather_3/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_3/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :°
tf.compat.v1.gather_3/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_3/GatherV2/indices:output:0,tf.compat.v1.gather_3/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         h
&tf.compat.v1.gather_2/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_2/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :°
tf.compat.v1.gather_2/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_2/GatherV2/indices:output:0,tf.compat.v1.gather_2/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         h
&tf.compat.v1.gather_5/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_5/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :°
tf.compat.v1.gather_5/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_5/GatherV2/indices:output:0,tf.compat.v1.gather_5/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         h
&tf.compat.v1.gather_4/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_4/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :°
tf.compat.v1.gather_4/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_4/GatherV2/indices:output:0,tf.compat.v1.gather_4/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         g
%tf.broadcast_to_1/BroadcastTo/shape/1Const*
_output_shapes
: *
dtype0*
value	B :g
%tf.broadcast_to_1/BroadcastTo/shape/2Const*
_output_shapes
: *
dtype0*
value	B :р
#tf.broadcast_to_1/BroadcastTo/shapePack%tf.compat.v1.floor_div_1/FloorDiv:z:0.tf.broadcast_to_1/BroadcastTo/shape/1:output:0.tf.broadcast_to_1/BroadcastTo/shape/2:output:0*
N*
T0*
_output_shapes
:╢
tf.broadcast_to_1/BroadcastToBroadcastTo$tf.expand_dims_1/ExpandDims:output:0,tf.broadcast_to_1/BroadcastTo/shape:output:0*
T0*+
_output_shapes
:         ]
tf.math.multiply_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ы
tf.math.multiply_2/MulMul'tf.compat.v1.gather_2/GatherV2:output:0!tf.math.multiply_2/Mul/y:output:0*
T0*'
_output_shapes
:         ]
tf.math.multiply_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ы
tf.math.multiply_3/MulMul'tf.compat.v1.gather_3/GatherV2:output:0!tf.math.multiply_3/Mul/y:output:0*
T0*'
_output_shapes
:         ]
tf.math.multiply_4/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ы
tf.math.multiply_4/MulMul'tf.compat.v1.gather_4/GatherV2:output:0!tf.math.multiply_4/Mul/y:output:0*
T0*'
_output_shapes
:         ]
tf.math.multiply_5/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ы
tf.math.multiply_5/MulMul'tf.compat.v1.gather_5/GatherV2:output:0!tf.math.multiply_5/Mul/y:output:0*
T0*'
_output_shapes
:         Х
tf.math.subtract_2/SubSubtf.math.subtract/Sub:z:0&tf.broadcast_to_1/BroadcastTo:output:0*
T0*+
_output_shapes
:         П
tf.__operators__.add_2/AddV2AddV2tf.math.multiply_4/Mul:z:0tf.math.multiply_5/Mul:z:0*
T0*'
_output_shapes
:         П
tf.__operators__.add_1/AddV2AddV2tf.math.multiply_2/Mul:z:0tf.math.multiply_3/Mul:z:0*
T0*'
_output_shapes
:         h
&tf.compat.v1.gather_8/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B : e
#tf.compat.v1.gather_8/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ё
tf.compat.v1.gather_8/GatherV2GatherV2tf.math.subtract_2/Sub:z:0/tf.compat.v1.gather_8/GatherV2/indices:output:0,tf.compat.v1.gather_8/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes

:У
tf.math.subtract_1/SubSub tf.__operators__.add_2/AddV2:z:0 tf.__operators__.add_1/AddV2:z:0*
T0*'
_output_shapes
:         Ю
tf.compat.v1.norm_1/norm/mulMul'tf.compat.v1.gather_8/GatherV2:output:0'tf.compat.v1.gather_8/GatherV2:output:0*
T0*
_output_shapes

:x
.tf.compat.v1.norm_1/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ╕
tf.compat.v1.norm_1/norm/SumSum tf.compat.v1.norm_1/norm/mul:z:07tf.compat.v1.norm_1/norm/Sum/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(u
tf.compat.v1.norm_1/norm/SqrtSqrt%tf.compat.v1.norm_1/norm/Sum:output:0*
T0*
_output_shapes

:К
 tf.compat.v1.norm_1/norm/SqueezeSqueeze!tf.compat.v1.norm_1/norm/Sqrt:y:0*
T0*
_output_shapes
:*
squeeze_dims
 Л
tf.compat.v1.norm/norm/mulMultf.math.subtract_1/Sub:z:0tf.math.subtract_1/Sub:z:0*
T0*'
_output_shapes
:         m
tf.compat.v1.norm/norm/ConstConst*
_output_shapes
:*
dtype0*
valueB"       в
tf.compat.v1.norm/norm/SumSumtf.compat.v1.norm/norm/mul:z:0%tf.compat.v1.norm/norm/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(q
tf.compat.v1.norm/norm/SqrtSqrt#tf.compat.v1.norm/norm/Sum:output:0*
T0*
_output_shapes

:k
tf.compat.v1.norm/norm/SqueezeSqueezetf.compat.v1.norm/norm/Sqrt:y:0*
T0*
_output_shapes
: ]
tf.math.multiply_8/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @К
tf.math.multiply_8/MulMul'tf.compat.v1.norm/norm/Squeeze:output:0!tf.math.multiply_8/Mul/y:output:0*
T0*
_output_shapes
: b
tf.math.reduce_max/ConstConst*
_output_shapes
:*
dtype0*
valueB: М
tf.math.reduce_max/MaxMax)tf.compat.v1.norm_1/norm/Squeeze:output:0!tf.math.reduce_max/Const:output:0*
T0*
_output_shapes
: А
tf.math.maximum/MaximumMaximumtf.math.multiply_8/Mul:z:0tf.math.reduce_max/Max:output:0*
T0*
_output_shapes
: П
tf.math.truediv/truedivRealDivtf.math.subtract/Sub:z:0tf.math.maximum/Maximum:z:0*
T0*+
_output_shapes
:         ═
flatten_1/PartitionedCallPartitionedCalltf.math.truediv/truediv:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_6670·
dense/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0
dense_7143
dense_7145*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         Z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_6683Ж
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_7148dense_1_7150*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         (*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_6700И
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_7153dense_2_7155*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_6717w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         к
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         ': : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:P L
'
_output_shapes
:         '
!
_user_specified_name	input_1
вy
є
?__inference_model_layer_call_and_return_conditional_losses_7050
input_1

dense_7034:Z

dense_7036:Z
dense_1_7039:Z(
dense_1_7041:(
dense_2_7044:(
dense_2_7046:
identityИвdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCallвdense_2/StatefulPartitionedCall╜
reshape_1/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_reshape_1_layer_call_and_return_conditional_losses_6574Б
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            Г
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           Г
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ш
&tf.__operators__.getitem/strided_sliceStridedSlice"reshape_1/PartitionedCall:output:05tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         *

begin_mask*
end_maskh
&tf.compat.v1.gather_1/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :П
tf.compat.v1.gather_1/GatherV2GatherV2/tf.__operators__.getitem/strided_slice:output:0/tf.compat.v1.gather_1/GatherV2/indices:output:0,tf.compat.v1.gather_1/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         f
$tf.compat.v1.gather/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :c
!tf.compat.v1.gather/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :Й
tf.compat.v1.gather/GatherV2GatherV2/tf.__operators__.getitem/strided_slice:output:0-tf.compat.v1.gather/GatherV2/indices:output:0*tf.compat.v1.gather/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         [
tf.math.multiply/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Х
tf.math.multiply/MulMul%tf.compat.v1.gather/GatherV2:output:0tf.math.multiply/Mul/y:output:0*
T0*'
_output_shapes
:         ]
tf.math.multiply_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ы
tf.math.multiply_1/MulMul'tf.compat.v1.gather_1/GatherV2:output:0!tf.math.multiply_1/Mul/y:output:0*
T0*'
_output_shapes
:         p
tf.compat.v1.size/SizeSize/tf.__operators__.getitem/strided_slice:output:0*
T0*
_output_shapes
: Л
tf.__operators__.add/AddV2AddV2tf.math.multiply/Mul:z:0tf.math.multiply_1/Mul:z:0*
T0*'
_output_shapes
:         _
tf.expand_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :е
tf.expand_dims/ExpandDims
ExpandDimstf.__operators__.add/AddV2:z:0&tf.expand_dims/ExpandDims/dim:output:0*
T0*+
_output_shapes
:         c
!tf.compat.v1.floor_div/FloorDiv/yConst*
_output_shapes
: *
dtype0*
value	B :Щ
tf.compat.v1.floor_div/FloorDivFloorDivtf.compat.v1.size/Size:output:0*tf.compat.v1.floor_div/FloorDiv/y:output:0*
T0*
_output_shapes
: e
#tf.broadcast_to/BroadcastTo/shape/1Const*
_output_shapes
: *
dtype0*
value	B :e
#tf.broadcast_to/BroadcastTo/shape/2Const*
_output_shapes
: *
dtype0*
value	B :╪
!tf.broadcast_to/BroadcastTo/shapePack#tf.compat.v1.floor_div/FloorDiv:z:0,tf.broadcast_to/BroadcastTo/shape/1:output:0,tf.broadcast_to/BroadcastTo/shape/2:output:0*
N*
T0*
_output_shapes
:░
tf.broadcast_to/BroadcastToBroadcastTo"tf.expand_dims/ExpandDims:output:0*tf.broadcast_to/BroadcastTo/shape:output:0*
T0*+
_output_shapes
:         и
tf.math.subtract/SubSub/tf.__operators__.getitem/strided_slice:output:0$tf.broadcast_to/BroadcastTo:output:0*
T0*+
_output_shapes
:         h
&tf.compat.v1.gather_7/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_7/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :°
tf.compat.v1.gather_7/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_7/GatherV2/indices:output:0,tf.compat.v1.gather_7/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         h
&tf.compat.v1.gather_6/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_6/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :°
tf.compat.v1.gather_6/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_6/GatherV2/indices:output:0,tf.compat.v1.gather_6/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         ]
tf.math.multiply_6/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ы
tf.math.multiply_6/MulMul'tf.compat.v1.gather_6/GatherV2:output:0!tf.math.multiply_6/Mul/y:output:0*
T0*'
_output_shapes
:         ]
tf.math.multiply_7/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ы
tf.math.multiply_7/MulMul'tf.compat.v1.gather_7/GatherV2:output:0!tf.math.multiply_7/Mul/y:output:0*
T0*'
_output_shapes
:         [
tf.compat.v1.size_1/SizeSizetf.math.subtract/Sub:z:0*
T0*
_output_shapes
: П
tf.__operators__.add_3/AddV2AddV2tf.math.multiply_6/Mul:z:0tf.math.multiply_7/Mul:z:0*
T0*'
_output_shapes
:         a
tf.expand_dims_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :л
tf.expand_dims_1/ExpandDims
ExpandDims tf.__operators__.add_3/AddV2:z:0(tf.expand_dims_1/ExpandDims/dim:output:0*
T0*+
_output_shapes
:         e
#tf.compat.v1.floor_div_1/FloorDiv/yConst*
_output_shapes
: *
dtype0*
value	B :Я
!tf.compat.v1.floor_div_1/FloorDivFloorDiv!tf.compat.v1.size_1/Size:output:0,tf.compat.v1.floor_div_1/FloorDiv/y:output:0*
T0*
_output_shapes
: h
&tf.compat.v1.gather_3/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_3/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :°
tf.compat.v1.gather_3/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_3/GatherV2/indices:output:0,tf.compat.v1.gather_3/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         h
&tf.compat.v1.gather_2/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_2/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :°
tf.compat.v1.gather_2/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_2/GatherV2/indices:output:0,tf.compat.v1.gather_2/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         h
&tf.compat.v1.gather_5/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_5/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :°
tf.compat.v1.gather_5/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_5/GatherV2/indices:output:0,tf.compat.v1.gather_5/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         h
&tf.compat.v1.gather_4/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_4/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :°
tf.compat.v1.gather_4/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_4/GatherV2/indices:output:0,tf.compat.v1.gather_4/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         g
%tf.broadcast_to_1/BroadcastTo/shape/1Const*
_output_shapes
: *
dtype0*
value	B :g
%tf.broadcast_to_1/BroadcastTo/shape/2Const*
_output_shapes
: *
dtype0*
value	B :р
#tf.broadcast_to_1/BroadcastTo/shapePack%tf.compat.v1.floor_div_1/FloorDiv:z:0.tf.broadcast_to_1/BroadcastTo/shape/1:output:0.tf.broadcast_to_1/BroadcastTo/shape/2:output:0*
N*
T0*
_output_shapes
:╢
tf.broadcast_to_1/BroadcastToBroadcastTo$tf.expand_dims_1/ExpandDims:output:0,tf.broadcast_to_1/BroadcastTo/shape:output:0*
T0*+
_output_shapes
:         ]
tf.math.multiply_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ы
tf.math.multiply_2/MulMul'tf.compat.v1.gather_2/GatherV2:output:0!tf.math.multiply_2/Mul/y:output:0*
T0*'
_output_shapes
:         ]
tf.math.multiply_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ы
tf.math.multiply_3/MulMul'tf.compat.v1.gather_3/GatherV2:output:0!tf.math.multiply_3/Mul/y:output:0*
T0*'
_output_shapes
:         ]
tf.math.multiply_4/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ы
tf.math.multiply_4/MulMul'tf.compat.v1.gather_4/GatherV2:output:0!tf.math.multiply_4/Mul/y:output:0*
T0*'
_output_shapes
:         ]
tf.math.multiply_5/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ы
tf.math.multiply_5/MulMul'tf.compat.v1.gather_5/GatherV2:output:0!tf.math.multiply_5/Mul/y:output:0*
T0*'
_output_shapes
:         Х
tf.math.subtract_2/SubSubtf.math.subtract/Sub:z:0&tf.broadcast_to_1/BroadcastTo:output:0*
T0*+
_output_shapes
:         П
tf.__operators__.add_2/AddV2AddV2tf.math.multiply_4/Mul:z:0tf.math.multiply_5/Mul:z:0*
T0*'
_output_shapes
:         П
tf.__operators__.add_1/AddV2AddV2tf.math.multiply_2/Mul:z:0tf.math.multiply_3/Mul:z:0*
T0*'
_output_shapes
:         h
&tf.compat.v1.gather_8/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B : e
#tf.compat.v1.gather_8/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ё
tf.compat.v1.gather_8/GatherV2GatherV2tf.math.subtract_2/Sub:z:0/tf.compat.v1.gather_8/GatherV2/indices:output:0,tf.compat.v1.gather_8/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes

:У
tf.math.subtract_1/SubSub tf.__operators__.add_2/AddV2:z:0 tf.__operators__.add_1/AddV2:z:0*
T0*'
_output_shapes
:         Ю
tf.compat.v1.norm_1/norm/mulMul'tf.compat.v1.gather_8/GatherV2:output:0'tf.compat.v1.gather_8/GatherV2:output:0*
T0*
_output_shapes

:x
.tf.compat.v1.norm_1/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ╕
tf.compat.v1.norm_1/norm/SumSum tf.compat.v1.norm_1/norm/mul:z:07tf.compat.v1.norm_1/norm/Sum/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(u
tf.compat.v1.norm_1/norm/SqrtSqrt%tf.compat.v1.norm_1/norm/Sum:output:0*
T0*
_output_shapes

:К
 tf.compat.v1.norm_1/norm/SqueezeSqueeze!tf.compat.v1.norm_1/norm/Sqrt:y:0*
T0*
_output_shapes
:*
squeeze_dims
 Л
tf.compat.v1.norm/norm/mulMultf.math.subtract_1/Sub:z:0tf.math.subtract_1/Sub:z:0*
T0*'
_output_shapes
:         m
tf.compat.v1.norm/norm/ConstConst*
_output_shapes
:*
dtype0*
valueB"       в
tf.compat.v1.norm/norm/SumSumtf.compat.v1.norm/norm/mul:z:0%tf.compat.v1.norm/norm/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(q
tf.compat.v1.norm/norm/SqrtSqrt#tf.compat.v1.norm/norm/Sum:output:0*
T0*
_output_shapes

:k
tf.compat.v1.norm/norm/SqueezeSqueezetf.compat.v1.norm/norm/Sqrt:y:0*
T0*
_output_shapes
: ]
tf.math.multiply_8/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @К
tf.math.multiply_8/MulMul'tf.compat.v1.norm/norm/Squeeze:output:0!tf.math.multiply_8/Mul/y:output:0*
T0*
_output_shapes
: b
tf.math.reduce_max/ConstConst*
_output_shapes
:*
dtype0*
valueB: М
tf.math.reduce_max/MaxMax)tf.compat.v1.norm_1/norm/Squeeze:output:0!tf.math.reduce_max/Const:output:0*
T0*
_output_shapes
: А
tf.math.maximum/MaximumMaximumtf.math.multiply_8/Mul:z:0tf.math.reduce_max/Max:output:0*
T0*
_output_shapes
: П
tf.math.truediv/truedivRealDivtf.math.subtract/Sub:z:0tf.math.maximum/Maximum:z:0*
T0*+
_output_shapes
:         ═
flatten_1/PartitionedCallPartitionedCalltf.math.truediv/truediv:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_6670·
dense/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0
dense_7034
dense_7036*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         Z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_6683Ж
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_7039dense_1_7041*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         (*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_6700И
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_7044dense_2_7046*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_6717w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         к
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         ': : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:P L
'
_output_shapes
:         '
!
_user_specified_name	input_1
їИ
ь
?__inference_model_layer_call_and_return_conditional_losses_7338

inputs6
$dense_matmul_readvariableop_resource:Z3
%dense_biasadd_readvariableop_resource:Z8
&dense_1_matmul_readvariableop_resource:Z(5
'dense_1_biasadd_readvariableop_resource:(8
&dense_2_matmul_readvariableop_resource:(5
'dense_2_biasadd_readvariableop_resource:
identityИвdense/BiasAdd/ReadVariableOpвdense/MatMul/ReadVariableOpвdense_1/BiasAdd/ReadVariableOpвdense_1/MatMul/ReadVariableOpвdense_2/BiasAdd/ReadVariableOpвdense_2/MatMul/ReadVariableOpE
reshape_1/ShapeShapeinputs*
T0*
_output_shapes
:g
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :╖
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:|
reshape_1/ReshapeReshapeinputs reshape_1/Reshape/shape:output:0*
T0*+
_output_shapes
:         Б
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            Г
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           Г
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         р
&tf.__operators__.getitem/strided_sliceStridedSlicereshape_1/Reshape:output:05tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         *

begin_mask*
end_maskh
&tf.compat.v1.gather_1/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :П
tf.compat.v1.gather_1/GatherV2GatherV2/tf.__operators__.getitem/strided_slice:output:0/tf.compat.v1.gather_1/GatherV2/indices:output:0,tf.compat.v1.gather_1/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         f
$tf.compat.v1.gather/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :c
!tf.compat.v1.gather/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :Й
tf.compat.v1.gather/GatherV2GatherV2/tf.__operators__.getitem/strided_slice:output:0-tf.compat.v1.gather/GatherV2/indices:output:0*tf.compat.v1.gather/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         [
tf.math.multiply/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Х
tf.math.multiply/MulMul%tf.compat.v1.gather/GatherV2:output:0tf.math.multiply/Mul/y:output:0*
T0*'
_output_shapes
:         ]
tf.math.multiply_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ы
tf.math.multiply_1/MulMul'tf.compat.v1.gather_1/GatherV2:output:0!tf.math.multiply_1/Mul/y:output:0*
T0*'
_output_shapes
:         p
tf.compat.v1.size/SizeSize/tf.__operators__.getitem/strided_slice:output:0*
T0*
_output_shapes
: Л
tf.__operators__.add/AddV2AddV2tf.math.multiply/Mul:z:0tf.math.multiply_1/Mul:z:0*
T0*'
_output_shapes
:         _
tf.expand_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :е
tf.expand_dims/ExpandDims
ExpandDimstf.__operators__.add/AddV2:z:0&tf.expand_dims/ExpandDims/dim:output:0*
T0*+
_output_shapes
:         c
!tf.compat.v1.floor_div/FloorDiv/yConst*
_output_shapes
: *
dtype0*
value	B :Щ
tf.compat.v1.floor_div/FloorDivFloorDivtf.compat.v1.size/Size:output:0*tf.compat.v1.floor_div/FloorDiv/y:output:0*
T0*
_output_shapes
: e
#tf.broadcast_to/BroadcastTo/shape/1Const*
_output_shapes
: *
dtype0*
value	B :e
#tf.broadcast_to/BroadcastTo/shape/2Const*
_output_shapes
: *
dtype0*
value	B :╪
!tf.broadcast_to/BroadcastTo/shapePack#tf.compat.v1.floor_div/FloorDiv:z:0,tf.broadcast_to/BroadcastTo/shape/1:output:0,tf.broadcast_to/BroadcastTo/shape/2:output:0*
N*
T0*
_output_shapes
:░
tf.broadcast_to/BroadcastToBroadcastTo"tf.expand_dims/ExpandDims:output:0*tf.broadcast_to/BroadcastTo/shape:output:0*
T0*+
_output_shapes
:         и
tf.math.subtract/SubSub/tf.__operators__.getitem/strided_slice:output:0$tf.broadcast_to/BroadcastTo:output:0*
T0*+
_output_shapes
:         h
&tf.compat.v1.gather_7/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_7/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :°
tf.compat.v1.gather_7/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_7/GatherV2/indices:output:0,tf.compat.v1.gather_7/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         h
&tf.compat.v1.gather_6/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_6/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :°
tf.compat.v1.gather_6/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_6/GatherV2/indices:output:0,tf.compat.v1.gather_6/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         ]
tf.math.multiply_6/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ы
tf.math.multiply_6/MulMul'tf.compat.v1.gather_6/GatherV2:output:0!tf.math.multiply_6/Mul/y:output:0*
T0*'
_output_shapes
:         ]
tf.math.multiply_7/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ы
tf.math.multiply_7/MulMul'tf.compat.v1.gather_7/GatherV2:output:0!tf.math.multiply_7/Mul/y:output:0*
T0*'
_output_shapes
:         [
tf.compat.v1.size_1/SizeSizetf.math.subtract/Sub:z:0*
T0*
_output_shapes
: П
tf.__operators__.add_3/AddV2AddV2tf.math.multiply_6/Mul:z:0tf.math.multiply_7/Mul:z:0*
T0*'
_output_shapes
:         a
tf.expand_dims_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :л
tf.expand_dims_1/ExpandDims
ExpandDims tf.__operators__.add_3/AddV2:z:0(tf.expand_dims_1/ExpandDims/dim:output:0*
T0*+
_output_shapes
:         e
#tf.compat.v1.floor_div_1/FloorDiv/yConst*
_output_shapes
: *
dtype0*
value	B :Я
!tf.compat.v1.floor_div_1/FloorDivFloorDiv!tf.compat.v1.size_1/Size:output:0,tf.compat.v1.floor_div_1/FloorDiv/y:output:0*
T0*
_output_shapes
: h
&tf.compat.v1.gather_3/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_3/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :°
tf.compat.v1.gather_3/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_3/GatherV2/indices:output:0,tf.compat.v1.gather_3/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         h
&tf.compat.v1.gather_2/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_2/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :°
tf.compat.v1.gather_2/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_2/GatherV2/indices:output:0,tf.compat.v1.gather_2/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         h
&tf.compat.v1.gather_5/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_5/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :°
tf.compat.v1.gather_5/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_5/GatherV2/indices:output:0,tf.compat.v1.gather_5/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         h
&tf.compat.v1.gather_4/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_4/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :°
tf.compat.v1.gather_4/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_4/GatherV2/indices:output:0,tf.compat.v1.gather_4/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:         g
%tf.broadcast_to_1/BroadcastTo/shape/1Const*
_output_shapes
: *
dtype0*
value	B :g
%tf.broadcast_to_1/BroadcastTo/shape/2Const*
_output_shapes
: *
dtype0*
value	B :р
#tf.broadcast_to_1/BroadcastTo/shapePack%tf.compat.v1.floor_div_1/FloorDiv:z:0.tf.broadcast_to_1/BroadcastTo/shape/1:output:0.tf.broadcast_to_1/BroadcastTo/shape/2:output:0*
N*
T0*
_output_shapes
:╢
tf.broadcast_to_1/BroadcastToBroadcastTo$tf.expand_dims_1/ExpandDims:output:0,tf.broadcast_to_1/BroadcastTo/shape:output:0*
T0*+
_output_shapes
:         ]
tf.math.multiply_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ы
tf.math.multiply_2/MulMul'tf.compat.v1.gather_2/GatherV2:output:0!tf.math.multiply_2/Mul/y:output:0*
T0*'
_output_shapes
:         ]
tf.math.multiply_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ы
tf.math.multiply_3/MulMul'tf.compat.v1.gather_3/GatherV2:output:0!tf.math.multiply_3/Mul/y:output:0*
T0*'
_output_shapes
:         ]
tf.math.multiply_4/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ы
tf.math.multiply_4/MulMul'tf.compat.v1.gather_4/GatherV2:output:0!tf.math.multiply_4/Mul/y:output:0*
T0*'
_output_shapes
:         ]
tf.math.multiply_5/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ы
tf.math.multiply_5/MulMul'tf.compat.v1.gather_5/GatherV2:output:0!tf.math.multiply_5/Mul/y:output:0*
T0*'
_output_shapes
:         Х
tf.math.subtract_2/SubSubtf.math.subtract/Sub:z:0&tf.broadcast_to_1/BroadcastTo:output:0*
T0*+
_output_shapes
:         П
tf.__operators__.add_2/AddV2AddV2tf.math.multiply_4/Mul:z:0tf.math.multiply_5/Mul:z:0*
T0*'
_output_shapes
:         П
tf.__operators__.add_1/AddV2AddV2tf.math.multiply_2/Mul:z:0tf.math.multiply_3/Mul:z:0*
T0*'
_output_shapes
:         h
&tf.compat.v1.gather_8/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B : e
#tf.compat.v1.gather_8/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ё
tf.compat.v1.gather_8/GatherV2GatherV2tf.math.subtract_2/Sub:z:0/tf.compat.v1.gather_8/GatherV2/indices:output:0,tf.compat.v1.gather_8/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes

:У
tf.math.subtract_1/SubSub tf.__operators__.add_2/AddV2:z:0 tf.__operators__.add_1/AddV2:z:0*
T0*'
_output_shapes
:         Ю
tf.compat.v1.norm_1/norm/mulMul'tf.compat.v1.gather_8/GatherV2:output:0'tf.compat.v1.gather_8/GatherV2:output:0*
T0*
_output_shapes

:x
.tf.compat.v1.norm_1/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ╕
tf.compat.v1.norm_1/norm/SumSum tf.compat.v1.norm_1/norm/mul:z:07tf.compat.v1.norm_1/norm/Sum/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(u
tf.compat.v1.norm_1/norm/SqrtSqrt%tf.compat.v1.norm_1/norm/Sum:output:0*
T0*
_output_shapes

:К
 tf.compat.v1.norm_1/norm/SqueezeSqueeze!tf.compat.v1.norm_1/norm/Sqrt:y:0*
T0*
_output_shapes
:*
squeeze_dims
 Л
tf.compat.v1.norm/norm/mulMultf.math.subtract_1/Sub:z:0tf.math.subtract_1/Sub:z:0*
T0*'
_output_shapes
:         m
tf.compat.v1.norm/norm/ConstConst*
_output_shapes
:*
dtype0*
valueB"       в
tf.compat.v1.norm/norm/SumSumtf.compat.v1.norm/norm/mul:z:0%tf.compat.v1.norm/norm/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(q
tf.compat.v1.norm/norm/SqrtSqrt#tf.compat.v1.norm/norm/Sum:output:0*
T0*
_output_shapes

:k
tf.compat.v1.norm/norm/SqueezeSqueezetf.compat.v1.norm/norm/Sqrt:y:0*
T0*
_output_shapes
: ]
tf.math.multiply_8/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @К
tf.math.multiply_8/MulMul'tf.compat.v1.norm/norm/Squeeze:output:0!tf.math.multiply_8/Mul/y:output:0*
T0*
_output_shapes
: b
tf.math.reduce_max/ConstConst*
_output_shapes
:*
dtype0*
valueB: М
tf.math.reduce_max/MaxMax)tf.compat.v1.norm_1/norm/Squeeze:output:0!tf.math.reduce_max/Const:output:0*
T0*
_output_shapes
: А
tf.math.maximum/MaximumMaximumtf.math.multiply_8/Mul:z:0tf.math.reduce_max/Max:output:0*
T0*
_output_shapes
: П
tf.math.truediv/truedivRealDivtf.math.subtract/Sub:z:0tf.math.maximum/Maximum:z:0*
T0*+
_output_shapes
:         `
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Е
flatten_1/ReshapeReshapetf.math.truediv/truediv:z:0flatten_1/Const:output:0*
T0*'
_output_shapes
:         А
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:Z*
dtype0Й
dense/MatMulMatMulflatten_1/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Z~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:Z*
dtype0И
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Z^
dense/Relu6Relu6dense/BiasAdd:output:0*
T0*'
_output_shapes
:         ZД
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:Z(*
dtype0М
dense_1/MatMulMatMuldense/Relu6:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         (В
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0О
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         (b
dense_1/Relu6Relu6dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         (Д
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0О
dense_2/MatMulMatMuldense_1/Relu6:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         В
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         f
dense_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*'
_output_shapes
:         b
IdentityIdentitydense_2/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:         Е
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         ': : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:O K
'
_output_shapes
:         '
 
_user_specified_nameinputs
║2
Т	
__inference__traced_save_7640
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableop2
.savev2_adam_m_dense_kernel_read_readvariableop2
.savev2_adam_v_dense_kernel_read_readvariableop0
,savev2_adam_m_dense_bias_read_readvariableop0
,savev2_adam_v_dense_bias_read_readvariableop4
0savev2_adam_m_dense_1_kernel_read_readvariableop4
0savev2_adam_v_dense_1_kernel_read_readvariableop2
.savev2_adam_m_dense_1_bias_read_readvariableop2
.savev2_adam_v_dense_1_bias_read_readvariableop4
0savev2_adam_m_dense_2_kernel_read_readvariableop4
0savev2_adam_v_dense_2_kernel_read_readvariableop2
.savev2_adam_m_dense_2_bias_read_readvariableop2
.savev2_adam_v_dense_2_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const

identity_1ИвMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: О

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*╖	
valueн	Bк	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЫ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B ╛	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableop.savev2_adam_m_dense_kernel_read_readvariableop.savev2_adam_v_dense_kernel_read_readvariableop,savev2_adam_m_dense_bias_read_readvariableop,savev2_adam_v_dense_bias_read_readvariableop0savev2_adam_m_dense_1_kernel_read_readvariableop0savev2_adam_v_dense_1_kernel_read_readvariableop.savev2_adam_m_dense_1_bias_read_readvariableop.savev2_adam_v_dense_1_bias_read_readvariableop0savev2_adam_m_dense_2_kernel_read_readvariableop0savev2_adam_v_dense_2_kernel_read_readvariableop.savev2_adam_m_dense_2_bias_read_readvariableop.savev2_adam_v_dense_2_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *%
dtypes
2	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:│
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*▒
_input_shapesЯ
Ь: :Z:Z:Z(:(:(:: : :Z:Z:Z:Z:Z(:Z(:(:(:(:(::: : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:Z: 

_output_shapes
:Z:$ 

_output_shapes

:Z(: 

_output_shapes
:(:$ 

_output_shapes

:(: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :$	 

_output_shapes

:Z:$
 

_output_shapes

:Z: 

_output_shapes
:Z: 

_output_shapes
:Z:$ 

_output_shapes

:Z(:$ 

_output_shapes

:Z(: 

_output_shapes
:(: 

_output_shapes
:(:$ 

_output_shapes

:(:$ 

_output_shapes

:(: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: "Ж
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*к
serving_defaultЦ
;
input_10
serving_default_input_1:0         ';
dense_20
StatefulPartitionedCall:0         tensorflow/serving/predict:ыл
Э
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer-19
layer-20
layer-21
layer-22
layer-23
layer-24
layer-25
layer-26
layer-27
layer-28
layer-29
layer-30
 layer-31
!layer-32
"layer-33
#layer-34
$layer-35
%layer-36
&layer-37
'layer-38
(layer-39
)layer-40
*layer-41
+layer_with_weights-0
+layer-42
,layer_with_weights-1
,layer-43
-layer_with_weights-2
-layer-44
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses
4_default_save_signature
5	optimizer
6
signatures"
_tf_keras_network
"
_tf_keras_input_layer
е
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses"
_tf_keras_layer
(
=	keras_api"
_tf_keras_layer
(
>	keras_api"
_tf_keras_layer
(
?	keras_api"
_tf_keras_layer
(
@	keras_api"
_tf_keras_layer
(
A	keras_api"
_tf_keras_layer
(
B	keras_api"
_tf_keras_layer
(
C	keras_api"
_tf_keras_layer
(
D	keras_api"
_tf_keras_layer
(
E	keras_api"
_tf_keras_layer
(
F	keras_api"
_tf_keras_layer
(
G	keras_api"
_tf_keras_layer
(
H	keras_api"
_tf_keras_layer
(
I	keras_api"
_tf_keras_layer
(
J	keras_api"
_tf_keras_layer
(
K	keras_api"
_tf_keras_layer
(
L	keras_api"
_tf_keras_layer
(
M	keras_api"
_tf_keras_layer
(
N	keras_api"
_tf_keras_layer
(
O	keras_api"
_tf_keras_layer
(
P	keras_api"
_tf_keras_layer
(
Q	keras_api"
_tf_keras_layer
(
R	keras_api"
_tf_keras_layer
(
S	keras_api"
_tf_keras_layer
(
T	keras_api"
_tf_keras_layer
(
U	keras_api"
_tf_keras_layer
(
V	keras_api"
_tf_keras_layer
(
W	keras_api"
_tf_keras_layer
(
X	keras_api"
_tf_keras_layer
(
Y	keras_api"
_tf_keras_layer
(
Z	keras_api"
_tf_keras_layer
(
[	keras_api"
_tf_keras_layer
(
\	keras_api"
_tf_keras_layer
(
]	keras_api"
_tf_keras_layer
(
^	keras_api"
_tf_keras_layer
(
_	keras_api"
_tf_keras_layer
(
`	keras_api"
_tf_keras_layer
(
a	keras_api"
_tf_keras_layer
(
b	keras_api"
_tf_keras_layer
(
c	keras_api"
_tf_keras_layer
е
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses"
_tf_keras_layer
╗
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses

pkernel
qbias"
_tf_keras_layer
╗
r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses

xkernel
ybias"
_tf_keras_layer
╜
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses
Аkernel
	Бbias"
_tf_keras_layer
L
p0
q1
x2
y3
А4
Б5"
trackable_list_wrapper
L
p0
q1
x2
y3
А4
Б5"
trackable_list_wrapper
 "
trackable_list_wrapper
╧
Вnon_trainable_variables
Гlayers
Дmetrics
 Еlayer_regularization_losses
Жlayer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
4_default_save_signature
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
═
Зtrace_0
Иtrace_1
Йtrace_2
Кtrace_32┌
$__inference_model_layer_call_fn_6739
$__inference_model_layer_call_fn_7197
$__inference_model_layer_call_fn_7214
$__inference_model_layer_call_fn_6941┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЗtrace_0zИtrace_1zЙtrace_2zКtrace_3
╣
Лtrace_0
Мtrace_1
Нtrace_2
Оtrace_32╞
?__inference_model_layer_call_and_return_conditional_losses_7338
?__inference_model_layer_call_and_return_conditional_losses_7462
?__inference_model_layer_call_and_return_conditional_losses_7050
?__inference_model_layer_call_and_return_conditional_losses_7159┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЛtrace_0zМtrace_1zНtrace_2zОtrace_3
╩B╟
__inference__wrapped_model_6554input_1"Ш
С▓Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
г
П
_variables
Р_iterations
С_learning_rate
Т_index_dict
У
_momentums
Ф_velocities
Х_update_step_xla"
experimentalOptimizer
-
Цserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Чnon_trainable_variables
Шlayers
Щmetrics
 Ъlayer_regularization_losses
Ыlayer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
ю
Ьtrace_02╧
(__inference_reshape_1_layer_call_fn_7467в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЬtrace_0
Й
Эtrace_02ъ
C__inference_reshape_1_layer_call_and_return_conditional_losses_7480в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЭtrace_0
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Юnon_trainable_variables
Яlayers
аmetrics
 бlayer_regularization_losses
вlayer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
ю
гtrace_02╧
(__inference_flatten_1_layer_call_fn_7485в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zгtrace_0
Й
дtrace_02ъ
C__inference_flatten_1_layer_call_and_return_conditional_losses_7491в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zдtrace_0
.
p0
q1"
trackable_list_wrapper
.
p0
q1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
еnon_trainable_variables
жlayers
зmetrics
 иlayer_regularization_losses
йlayer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
ъ
кtrace_02╦
$__inference_dense_layer_call_fn_7500в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zкtrace_0
Е
лtrace_02ц
?__inference_dense_layer_call_and_return_conditional_losses_7511в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zлtrace_0
:Z2dense/kernel
:Z2
dense/bias
.
x0
y1"
trackable_list_wrapper
.
x0
y1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
мnon_trainable_variables
нlayers
оmetrics
 пlayer_regularization_losses
░layer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
ь
▒trace_02═
&__inference_dense_1_layer_call_fn_7520в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z▒trace_0
З
▓trace_02ш
A__inference_dense_1_layer_call_and_return_conditional_losses_7531в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z▓trace_0
 :Z(2dense_1/kernel
:(2dense_1/bias
0
А0
Б1"
trackable_list_wrapper
0
А0
Б1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
│non_trainable_variables
┤layers
╡metrics
 ╢layer_regularization_losses
╖layer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ь
╕trace_02═
&__inference_dense_2_layer_call_fn_7540в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╕trace_0
З
╣trace_02ш
A__inference_dense_2_layer_call_and_return_conditional_losses_7551в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╣trace_0
 :(2dense_2/kernel
:2dense_2/bias
 "
trackable_list_wrapper
■
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
+42
,43
-44"
trackable_list_wrapper
(
║0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЎBє
$__inference_model_layer_call_fn_6739input_1"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
їBЄ
$__inference_model_layer_call_fn_7197inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
їBЄ
$__inference_model_layer_call_fn_7214inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЎBє
$__inference_model_layer_call_fn_6941input_1"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
РBН
?__inference_model_layer_call_and_return_conditional_losses_7338inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
РBН
?__inference_model_layer_call_and_return_conditional_losses_7462inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
СBО
?__inference_model_layer_call_and_return_conditional_losses_7050input_1"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
СBО
?__inference_model_layer_call_and_return_conditional_losses_7159input_1"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Л
Р0
╗1
╝2
╜3
╛4
┐5
└6
┴7
┬8
├9
─10
┼11
╞12"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
P
╗0
╜1
┐2
┴3
├4
┼5"
trackable_list_wrapper
P
╝0
╛1
└2
┬3
─4
╞5"
trackable_list_wrapper
┐2╝╣
о▓к
FullArgSpec2
args*Ъ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
╔B╞
"__inference_signature_wrapper_7180input_1"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▄B┘
(__inference_reshape_1_layer_call_fn_7467inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
C__inference_reshape_1_layer_call_and_return_conditional_losses_7480inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▄B┘
(__inference_flatten_1_layer_call_fn_7485inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
C__inference_flatten_1_layer_call_and_return_conditional_losses_7491inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
╪B╒
$__inference_dense_layer_call_fn_7500inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
єBЁ
?__inference_dense_layer_call_and_return_conditional_losses_7511inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
┌B╫
&__inference_dense_1_layer_call_fn_7520inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
їBЄ
A__inference_dense_1_layer_call_and_return_conditional_losses_7531inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
┌B╫
&__inference_dense_2_layer_call_fn_7540inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
їBЄ
A__inference_dense_2_layer_call_and_return_conditional_losses_7551inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
R
╟	variables
╚	keras_api

╔total

╩count"
_tf_keras_metric
#:!Z2Adam/m/dense/kernel
#:!Z2Adam/v/dense/kernel
:Z2Adam/m/dense/bias
:Z2Adam/v/dense/bias
%:#Z(2Adam/m/dense_1/kernel
%:#Z(2Adam/v/dense_1/kernel
:(2Adam/m/dense_1/bias
:(2Adam/v/dense_1/bias
%:#(2Adam/m/dense_2/kernel
%:#(2Adam/v/dense_2/kernel
:2Adam/m/dense_2/bias
:2Adam/v/dense_2/bias
0
╔0
╩1"
trackable_list_wrapper
.
╟	variables"
_generic_user_object
:  (2total
:  (2countТ
__inference__wrapped_model_6554opqxyАБ0в-
&в#
!К
input_1         '
к "1к.
,
dense_2!К
dense_2         и
A__inference_dense_1_layer_call_and_return_conditional_losses_7531cxy/в,
%в"
 К
inputs         Z
к ",в)
"К
tensor_0         (
Ъ В
&__inference_dense_1_layer_call_fn_7520Xxy/в,
%в"
 К
inputs         Z
к "!К
unknown         (к
A__inference_dense_2_layer_call_and_return_conditional_losses_7551eАБ/в,
%в"
 К
inputs         (
к ",в)
"К
tensor_0         
Ъ Д
&__inference_dense_2_layer_call_fn_7540ZАБ/в,
%в"
 К
inputs         (
к "!К
unknown         ж
?__inference_dense_layer_call_and_return_conditional_losses_7511cpq/в,
%в"
 К
inputs         
к ",в)
"К
tensor_0         Z
Ъ А
$__inference_dense_layer_call_fn_7500Xpq/в,
%в"
 К
inputs         
к "!К
unknown         Zк
C__inference_flatten_1_layer_call_and_return_conditional_losses_7491c3в0
)в&
$К!
inputs         
к ",в)
"К
tensor_0         
Ъ Д
(__inference_flatten_1_layer_call_fn_7485X3в0
)в&
$К!
inputs         
к "!К
unknown         ╡
?__inference_model_layer_call_and_return_conditional_losses_7050rpqxyАБ8в5
.в+
!К
input_1         '
p 

 
к ",в)
"К
tensor_0         
Ъ ╡
?__inference_model_layer_call_and_return_conditional_losses_7159rpqxyАБ8в5
.в+
!К
input_1         '
p

 
к ",в)
"К
tensor_0         
Ъ ┤
?__inference_model_layer_call_and_return_conditional_losses_7338qpqxyАБ7в4
-в*
 К
inputs         '
p 

 
к ",в)
"К
tensor_0         
Ъ ┤
?__inference_model_layer_call_and_return_conditional_losses_7462qpqxyАБ7в4
-в*
 К
inputs         '
p

 
к ",в)
"К
tensor_0         
Ъ П
$__inference_model_layer_call_fn_6739gpqxyАБ8в5
.в+
!К
input_1         '
p 

 
к "!К
unknown         П
$__inference_model_layer_call_fn_6941gpqxyАБ8в5
.в+
!К
input_1         '
p

 
к "!К
unknown         О
$__inference_model_layer_call_fn_7197fpqxyАБ7в4
-в*
 К
inputs         '
p 

 
к "!К
unknown         О
$__inference_model_layer_call_fn_7214fpqxyАБ7в4
-в*
 К
inputs         '
p

 
к "!К
unknown         к
C__inference_reshape_1_layer_call_and_return_conditional_losses_7480c/в,
%в"
 К
inputs         '
к "0в-
&К#
tensor_0         
Ъ Д
(__inference_reshape_1_layer_call_fn_7467X/в,
%в"
 К
inputs         '
к "%К"
unknown         а
"__inference_signature_wrapper_7180zpqxyАБ;в8
в 
1к.
,
input_1!К
input_1         '"1к.
,
dense_2!К
dense_2         