#include <torch/csrc/jit/codegen/cuda/aggregate_dag.h>
#include <torch/csrc/jit/codegen/cuda/multigroup_fusion.h>


namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

AggregateExpr::AggregateExpr(
    IrBuilderPasskey passkey, GroupPtr group)
    : Expr(passkey, ExprType::AggregateExpr), group_(group) {
}

AggregateExpr::AggregateExpr(const AggregateExpr* src, IrCloner* ir_cloner)
    : Expr(src, ir_cloner), group_(src->group_) {}

Expr* AggregateExpr::shallowCopy() const {
  auto result = IrBuilder::create<AggregateExpr>(this->group_);
  result->copyPredicatesFrom(this);
  return result;
}

bool AggregateExpr::sameAs(const Statement* other) const {
  if (this == other) {
    return true;
  }
  if (!other->isA<AggregateExpr>()) {
    return false;
  }
  const auto other_op = other->as<AggregateExpr>();

  return group_== other_op->getGroup();
}


SendRecv::SendRecv(IrBuilderPasskey passkey, AggregateVal* out, AggregateVal* in)
    : Expr(passkey, ExprType::SendRecv), out_{out}, in_{in} {
  addOutput(out);
  addInput(in);
}

SendRecv::SendRecv(const SendRecv* src, IrCloner* ir_cloner)
    : Expr(src, ir_cloner), out_{src->out_}, in_{src->in_} {}

Expr* SendRecv::shallowCopy() const {
  auto result = IrBuilder::create<SendRecv>(this->out_, this->in_);
  result->copyPredicatesFrom(this);
  return result;
}

bool SendRecv::sameAs(const Statement* other) const {
  if (this == other) {
    return true;
  }
  if (!other->isA<SendRecv>()) {
    return false;
  }
  return Expr::sameAs(other);
}


AggregateVal::AggregateVal(
    IrBuilderPasskey passkey, Val* val, GroupPtr group)
    : Val(passkey, ValType::AggregateVal, val->dtype()),
    original_val_(val), group_(group) {}

AggregateVal::AggregateVal(const AggregateVal* src, IrCloner* ir_cloner)
    : Val(src, ir_cloner), original_val_(src->original_val_), group_(src->group_) {}

bool AggregateVal::sameAs(const Statement* other) const {
  if (this == other) {
    return true;
  }
  if (!other->isA<AggregateVal>()) {
    return false;
  }
  const auto other_aggregate_val = other->as<AggregateVal>();
  return original_val_->sameAs(other_aggregate_val->original_val_)
          && group_->unique_id == other_aggregate_val->group_->unique_id;
}


AggregateDag::AggregateDag():Fusion(), IterVisitor(){}

void AggregateDag::build(MultiGroupFusion* fusion) {
  // iterate over all groups in the multigroup fusion
  for (auto group: fusion->groups()) {
    // set the container we will add the IR to
    IrContainer* container = (IrContainer*)this;
    // temp container for the inputs of the current group
    std::vector<AggregateVal*> inputs;
    // temp container for the outputs of the current group
    std::vector<AggregateVal*> outputs;

    for (auto output_val : group->output_vals) {
      // creates the AggregateVal corresponding to each output of the current group
      auto val = IrBuilder::create<AggregateVal>(container, output_val, group);
      // tag current group as producer of this val
      producer[output_val] = val;
      // add the new IR to the outputs container
      outputs.push_back(val);
    }

    for (auto input_val : group->input_vals) {
      // creates the AggregateVal corresponding to each input of the current group
      auto val = IrBuilder::create<AggregateVal>(container, input_val, group);
      // add current group as a consumer of this val
      consumers.insert({input_val, val});
      // add the new IR to the inputs container
      inputs.push_back(val);
      if (producer.find(input_val) != producer.end()){
        // means that input_val is not a global input and 
        // so is produced by another group
        auto src = producer[input_val];
        // create and IR indicating the val needs to be sent between groups
        auto sendRecv = IrBuilder::create<SendRecv>(container, val, src);
        // the received val is set as output of sendRecv
        val->setDefinition(sendRecv);
      } else {
        //add val as global input of the aggregate dag
        addInput(val);
      }
    }

    // create IR representing the exprs contained in the current group
    // This could be moved to AggregateExpr's initializer.
    // Question: what does the following mean? "Constructors need to register with the Fusion after inputs/outputs are defined"
    auto expr = IrBuilder::create<AggregateExpr>(container, group);
    for (auto out: outputs){
      // set each output as an output of the group
      expr->addOutput(out);
      // set definition of the ouput
      out->setDefinition(expr);
    }
    for (auto in: inputs){
      // set each input as an input of the group
      expr->addInput(in);
    }
  }
  for (auto it: producer){
    if (consumers.count(it.first)==0){
      // means that it is a global output.
      // TODO: Im not sure it is exact, but it is not critical for now
        addOutput(it.second);
    }
  }
}


} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch

