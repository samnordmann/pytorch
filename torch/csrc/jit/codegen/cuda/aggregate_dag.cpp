#include <torch/csrc/jit/codegen/cuda/aggregate_dag.h>
#include <torch/csrc/jit/codegen/cuda/multidevice_runtime.h>
#include <torch/csrc/jit/codegen/cuda/ir_printer.h>


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
    : Val(passkey, ValType::AggregateVal, val->dtype()),
    original_val_(val), group_(group) {
      //TODO: add a mapping from original val to AggregateVal.
    }

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

AggregateDag::AggregateDag():IrContainer(){}

void AggregateDag::build(MultiGroupFusion* fusion) {
  for (auto group_ptr: fusion->groups()) {
    auto group = group_ptr.get();
    IrContainer* container = (IrContainer*)this;

    // auto expr = IrBuilder::create<AggregateExpr>(container, group);

    for (auto output_val : group->output_vals) {
      auto val = IrBuilder::create<AggregateVal>(container, output_val, group);
      // producer[output_val] = val;
      // expr->addOutput(val);
      // val->setDefinition(expr);
    }

    // for (auto input_val : group->input_vals) {
    //   auto val = IrBuilder::create<AggregateVal>(container, input_val, group);
    //   // consumers[input_val].pushBack(val);
    //   expr->addInput(val);
    //   if (producer.find(input_val) != producer.end()){
    //     //means that input_val is not a global input and 
    //     // is produced by another group
    //     auto src = producer[input_val];
    //     auto sendRecv = IrBuilder::create<SendRecv>(container);
    //     sendRecv->addInput(src);
    //     sendRecv->addOutput(val);
    //     val->setDefinition(sendRecv);
    //   }
    // }
  }
}




// std::ostream& operator<< (std::ostream &out, AggregateVal const& data) {
//     return out << "AggregateVal represents Val " <<  data.getOriginalVal()
//               << "on group " << data.getGroup() << "\n";
// }

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

  void AggregateDag::print(){
    IrPrinter p(std::cout);
    std::cout << "AggregateDag containing Vals {\n";
    for (auto val: vals())
      p.handle((AggregateVal*)val);
    std::cout << "}\n and Exprs {";
    for (auto expr: unordered_exprs())
      std::cout << expr;
    std::cout << "}\n" << std::endl;
  }




} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch

