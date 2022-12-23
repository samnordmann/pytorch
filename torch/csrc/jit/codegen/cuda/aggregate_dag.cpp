#include <torch/csrc/jit/codegen/cuda/aggregate_dag.h>
#include <torch/csrc/jit/codegen/cuda/multidevice_runtime.h>


namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

Expr* SendRecv::shallowCopy() const {
  auto result = IrBuilder::create<SendRecv>();
  result->copyPredicatesFrom(this);
  return result;
}

Expr* AggregateExpr::shallowCopy() const {
  auto result = IrBuilder::create<AggregateExpr>(this->group_);
  result->copyPredicatesFrom(this);
  return result;
}

AggregateExpr::AggregateExpr(
    IrBuilderPasskey passkey,
    Group* group)
    : Expr(passkey, ExprType::AggregateExpr), group_(group) {
  for (auto input: group_->input_vals){
    addInput(input);
  }
  for (auto output: group_->output_vals){
    addOutput(output);
  }
}

SendRecv::SendRecv(IrBuilderPasskey passkey)
    : Expr(passkey, ExprType::SendRecv) {}

AggregateVal::AggregateVal(
    IrBuilderPasskey passkey, Val* val, Group* group)
    : Val(passkey, val->vtype(), val->dtype()),
    original_val_(val), group_(group) {
      //TODO: add a mapping from original val to AggregateVal.
    }

AggregateDag::AggregateDag():IrContainer(){}

void AggregateDag::build(MultiGroupFusion* fusion) {
  for (auto group_ptr: fusion->groups()) {
    auto group = group_ptr.get();
    IrContainer* container = (IrContainer*)this;

    auto expr = IrBuilder::create<AggregateExpr>(container, group);

    for (auto output_val : group->output_vals) {
      auto val = IrBuilder::create<AggregateVal>(container, output_val, group);
      producer[output_val] = val;
      expr->addOutput(val);
      val->setDefinition(expr);
    }

    for (auto input_val : group->input_vals) {
      auto val = IrBuilder::create<AggregateVal>(container, input_val, group);
      // consumers[input_val].pushBack(val);
      expr->addInput(val);
      if (producer.find(input_val) != producer.end()){
        //means that input_val is not a global input and 
        // is produced by another group
        auto src = producer[input_val];
        auto sendRecv = IrBuilder::create<SendRecv>(container);
        sendRecv->addInput(src);
        sendRecv->addOutput(val);
        val->setDefinition(sendRecv);
      }
    }
  }
}




std::ostream& operator<< (std::ostream &out, AggregateVal const& data) {
    return out << "AggregateVal represents Val " <<  data.getOriginalVal()
              << "on group " << data.getGroup() << "\n";
}

std::ostream& operator<< (std::ostream &out, AggregateExpr const& data) {
    return out << "AggregateExpr represents Group " <<  data.getGroup() << "\n";
}

std::ostream& operator<< (std::ostream &out, SendRecv const& data) {
    out << "SendRecv with inputs ";
    for (auto input: data.inputs())
      out << input;
    out<< " and outputs ";
    for (auto output: data.outputs())
      out << output;
    out << "\n";
    return out;
}

std::ostream& operator<< (std::ostream &out, AggregateDag const& data) {
    out << "AggregateDag with Vals ";
    for (auto val: data.vals())
      out << val;
    out<< " and Exprs ";
    for (auto expr: data.unordered_exprs())
      out << expr;
    out << "\n";
    return out;
}





} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch

